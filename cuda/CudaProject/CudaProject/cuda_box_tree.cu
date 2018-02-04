#include "cuda_box_tree.hpp"

#include "cuda_runtime.h"

#include <cstdio>

__device__ ppc::cuda::box_tree* g_tree;

__global__ void insert_values_kernel(ppc::cuda::value_type* values, std::size_t size)
{
	if (!g_tree)
	{
		g_tree = new ppc::cuda::box_tree{};
	}

	for (std::size_t i = 0; i < size; ++i)
	{
		g_tree->insert(values[i]);
	}
}

__global__ void print_values_kernel(std::size_t* numOfItems)
{
	if (!g_tree)
	{
		std::printf("No values.\n");
		*numOfItems = 0;
	}
	else
	{
		*numOfItems = g_tree->size();

		/*std::printf("Box tree: ");
		for (const auto& values : *g_tree)
		{
			std::printf("%d ", values.second);
		}
		std::printf("\n");*/
	}

	delete g_tree;
}

//find_values_kernel<<<NUM_OF_BLOKS, NUM_OF_THREADS>>>(d_boxes, d_results, d_numOfResults);
__global__ void find_values_kernel(
	ppc::cuda::bbox* boxes, int numOfBoxes, 
	int* results, int* numOfResults)
{
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < numOfBoxes)
	{
		auto it = g_tree->find(boxes[index]);
		if (it != g_tree->end())
		{
			results[index] = it->second;
			atomicAdd(numOfResults, 1);
		}
		else
		{
			results[index] = -1;
		}
	}
}

namespace ppc
{
	namespace cuda
	{
		void insert_values(std::vector<value_type> values)
		{
			const auto size = sizeof(value_type) * values.size();
			value_type* d_values = nullptr;
			cudaMalloc(&d_values, size);
			cudaMemcpy(d_values, &values[0], size, cudaMemcpyHostToDevice);

			insert_values_kernel<<<1,1>>>(d_values, static_cast<int>(values.size()));

			cudaDeviceSynchronize();
			cudaFree(d_values);
		}

		void print_values()
		{
			std::size_t* d_numOfItems;
			cudaMalloc(&d_numOfItems, sizeof(std::size_t));

			print_values_kernel<<<1,1>>>(d_numOfItems);

			std::size_t numOfItems = 0;
			cudaMemcpy(&numOfItems, d_numOfItems, sizeof(std::size_t), cudaMemcpyDeviceToHost);
			printf("Number of items: %d\n", static_cast<int>(numOfItems));
		}

		std::vector<int> find_values(const std::vector<bbox>& boxes)
		{
			constexpr auto NUM_OF_THREADS_PER_BLOCK = 512;
			const auto numOfBlocks = static_cast<int>(std::ceil(boxes.size() / static_cast<double>(NUM_OF_THREADS_PER_BLOCK)));

			bbox* d_boxes;
			int* d_results;
			int* d_numOfResults;
			cudaMalloc(&d_boxes, sizeof(bbox) * boxes.size());
			cudaMalloc(&d_results, sizeof(int) * boxes.size());
			cudaMalloc(&d_numOfResults, sizeof(int));

			cudaMemcpy(d_boxes, &boxes[0], sizeof(bbox) * boxes.size(), cudaMemcpyHostToDevice);

			int zero = 0;
			cudaMemcpy(d_numOfResults, &zero, sizeof(int), cudaMemcpyHostToDevice);

			find_values_kernel<<<numOfBlocks, NUM_OF_THREADS_PER_BLOCK>>>(
				d_boxes, static_cast<int>(boxes.size()),
				d_results, d_numOfResults);

			int numOfResults = 0;
			cudaMemcpy(&numOfResults, d_numOfResults, sizeof(int), cudaMemcpyDeviceToHost);

			std::vector<int> results(numOfResults);
			cudaMemcpy(&results[0], d_results, sizeof(int) * numOfResults, cudaMemcpyDeviceToHost);

			cudaFree(d_boxes);
			cudaFree(d_results);
			cudaFree(d_numOfResults);

			return results;
		}
	}
}