#include "cuda_box_tree.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using bbox = ppc::box_tree::bbox;
using bboxes = ppc::box_tree::bboxes;
using value = ppc::box_tree::value;
using box_value_pair = ppc::box_tree::box_value_pair;
using bbox_tree = ppc::volume_tree<
	bbox,
	int,
	ppc::Intersect,
	ppc::Expand,
	std::less<bbox>,
	ppc::host_allocator
>;

__device__ bbox_tree* g_tree = nullptr;

__global__ void init_kernel()
{
	if (g_tree)
	{
		delete g_tree;
	}
	
	g_tree = new bbox_tree{};
}

__global__ void destroy_kernel()
{
	if (g_tree)
	{
		delete g_tree;
	}
}

__global__ void insert_kernel(box_value_pair* values, std::size_t numOfValues)
{
	for (std::size_t i = 0; i < numOfValues; ++i)
	{
		g_tree->insert(values[i]);
	}
}

__global__ void size_kernel(std::size_t* size)
{
	*size = g_tree->size();
}

__global__ void find_kernel(bbox* boxes, std::size_t numOfBoxes, value* results)
{
	//TODO: shared results?

	const auto index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numOfBoxes)
	{
		auto it = g_tree->find(boxes[index]);
		value val = -1;
		if (it != g_tree->end())
		{
			val = it->second;
		}

		//results[index] = val;
	}
}

namespace ppc
{
	namespace box_tree
	{
		cuda_box_tree::cuda_box_tree()
		{
			auto cudaStatus = cudaSetDevice(0);
			if (cudaStatus != cudaSuccess) 
			{
				std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
				throw std::runtime_error{ "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" };
			}

			auto printStackAndHeapSizes = [](std::string when)
			{
				std::size_t stackSize = 0;
				std::size_t heapSize = 0;
				cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
				cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
				std::cout << when << " stack size: " << stackSize << ", " << when << " heap size: " << heapSize << std::endl;
			};

			printStackAndHeapSizes("Current");

			auto error = cudaDeviceSetLimit(cudaLimitStackSize, 100 * 1024);
			if (error != cudaSuccess)
			{
				std::cerr << "Can't set the stack size!" << std::endl;
				throw std::runtime_error{ "Can't set the stack size!" };
			}

			printStackAndHeapSizes("New");

			init_kernel<<<1,1>>>();
			check_for_errors();
		}

		cuda_box_tree::~cuda_box_tree()
		{
			destroy_kernel<<<1,1>>>();
			check_for_errors();
		}

		void cuda_box_tree::insert(const box_value_pair& value)
		{
			insert(box_value_pairs{ value });
		}

		void cuda_box_tree::insert(const box_value_pairs& values)
		{
			const auto valuesSize = sizeof(box_value_pair) * values.size();

			box_value_pair* d_values{};
			cudaMalloc(&d_values, valuesSize);
			cudaMemcpy(d_values, &values[0], valuesSize, cudaMemcpyHostToDevice);

			insert_kernel<<<1,1>>>(d_values, values.size());
			check_for_errors();

			cudaFree(d_values);
		}

		std::size_t cuda_box_tree::size() const
		{
			std::size_t* d_size{};
			cudaMalloc(&d_size, sizeof(std::size_t));

			size_kernel<<<1,1>>>(d_size);
			check_for_errors();

			std::size_t treeSize{};
			cudaMemcpy(&treeSize, d_size, sizeof(std::size_t), cudaMemcpyDeviceToHost);

			cudaFree(d_size);
			return treeSize;
		}

		values cuda_box_tree::find(const bboxes& boxes) const
		{
			constexpr auto numOfThreads = 512;
			const auto numOfBlocks = static_cast<int>(std::ceil(boxes.size() / static_cast<double>(numOfThreads)));

			const auto boxesSize = sizeof(bbox) * boxes.size();
			const auto valuesSize = sizeof(value) * boxes.size();
			
			bbox* d_boxes{};
			cudaMalloc(&d_boxes, boxesSize);
			cudaMemcpy(d_boxes, &boxes[0], boxesSize, cudaMemcpyHostToDevice);

			value* d_values{};
			cudaMalloc(&d_values, valuesSize);

			find_kernel<<<numOfBlocks,numOfThreads>>>(d_boxes, boxes.size(), d_values);
			check_for_errors();

			values valsBuffer(boxes.size());
			cudaMemcpy(&valsBuffer[0], d_values, valuesSize, cudaMemcpyDeviceToHost);
			valsBuffer.erase(std::remove(valsBuffer.begin(), valsBuffer.end(), -1), valsBuffer.end());
			
			cudaFree(d_boxes);
			cudaFree(d_values);

			return valsBuffer;
		}

		void cuda_box_tree::check_for_errors() const
		{
			auto cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) 
			{
				std::cerr << "Cuda error: " << cudaGetErrorString(cudaStatus) << std::endl;
				throw std::runtime_error{ "Cuda error" };
			}
		}
	}
}