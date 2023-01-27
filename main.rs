#![allow(unused)]
use anyhow::{Context, Result};
use ash::{
    self,
    vk::{self, DeviceQueueCreateInfo},
};
use gpu_allocator::{vulkan::*, MemoryLocation};
use std::env;
use std::time;

fn main() -> Result<()> {
    // Config
    let mut args = env::args().skip(1);
    let width = args.next().context("width is required")?.parse::<u64>()?;
    let height = args.next().context("height is required")?.parse::<u64>()?;
    let red = args.next().context("red is required")?.parse::<u32>()?;
    let green = args.next().context("green is required")?.parse::<u32>()?;
    let blue = args.next().context("blue is required")?.parse::<u32>()?;
    let alpha = args.next().context("alpha is required")?.parse::<u32>()?;

    // Context
    let entry = unsafe { ash::Entry::load() }?;

    let instance = {
        let application_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::builder().application_info(&application_info);
        unsafe { entry.create_instance(&create_info, None) }?
    };

    let physical_device = unsafe { instance.enumerate_physical_devices() }?
        .into_iter()
        .next()
        .context("No physical device found")?;

    let device = {
        let queue_priorities = [1.0];
        let queue_create_info = DeviceQueueCreateInfo::builder()
            .queue_family_index(0)
            .queue_priorities(&queue_priorities);

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));
        unsafe { instance.create_device(physical_device, &create_info, None) }?
    };

    let queue = unsafe { device.get_device_queue(0, 0) };

    // Create allocator
    let mut allocator = {
        let allocator_create_description = AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
        };
        Allocator::new(&allocator_create_description)?
    };

    let value_count = width * height;
    // Create buffer
    let buffer = {
        let create_info = vk::BufferCreateInfo::builder()
            .size(value_count * std::mem::size_of::<u32>() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        unsafe { device.create_buffer(&create_info, None) }?
    };

    let allocation = {
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_create_description = AllocationCreateDesc {
            name: "Buffer allocation",
            requirements: memory_requirements,
            location: MemoryLocation::GpuToCpu,
            linear: true,
        };

        let allocation = allocator.allocate(&allocation_create_description)?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) };
        allocation
    };

    let command_pool = {
        let create_info = vk::CommandPoolCreateInfo::builder().queue_family_index(0);
        unsafe { device.create_command_pool(&create_info, None) }?
    };

    let command_buffer = {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);
        unsafe { device.allocate_command_buffers(&create_info) }?
            .into_iter()
            .next()
            .context("No command buffer found")?
    };

    // Recording command buffer
    {
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }?;
    }

    let value = red | green << 8 | blue << 16 | alpha << 24;
    unsafe {
        device.cmd_fill_buffer(
            command_buffer,
            buffer,
            allocation.offset(),
            allocation.size(),
            value,
        )
    }

    unsafe { device.end_command_buffer(command_buffer) }?;

    // Creating synchronization object (Fence)
    let fence = {
        let create_info = vk::FenceCreateInfo::builder().build();
        unsafe { device.create_fence(&create_info, None) }?
    };

    // Execute command buffer by uploading it to the GPU through the queue
    {
        let submit_info =
            vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe { device.queue_submit(queue, std::slice::from_ref(&submit_info), fence) };
    }

    // Wait for the execution to complete
    let start = time::Instant::now();
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX) };
    println!("GPU took {:?}", time::Instant::now() - start);

    let data = allocation
        .mapped_slice()
        .context("Cannot access buffer from Host")?;

    let start = time::Instant::now();
    image::save_buffer(
        "image.png",
        data,
        width as u32,
        height as u32,
        image::ColorType::Rgba8,
    );
    println!("Saving took {:?}", time::Instant::now() - start);

    // Cleanup
    unsafe { device.destroy_fence(fence, None) }
    unsafe { device.destroy_command_pool(command_pool, None) }

    allocator.free(allocation)?;
    drop(allocator);
    unsafe { device.destroy_buffer(buffer, None) }

    unsafe { device.destroy_device(None) }
    unsafe { instance.destroy_instance(None) }
    Ok(())
}
