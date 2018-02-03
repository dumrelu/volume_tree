
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "bounding_box.hpp"
#include "volume_tree.hpp"
#include "host_allocator.hpp"

using bbox = ppc::bounding_box<int>;
using point = bbox::point_type;
using box_tree = ppc::volume_tree<
	bbox,
	int,
	ppc::Intersect,
	ppc::Expand,
	std::less<bbox>,
	ppc::host_allocator
>;

__global__ void treeKernel(box_tree* btree, int* dummy)
{
	//box_tree tree;
	btree->insert({ { { 0, 0, 0 },{ 10, 10, 10 } }, 1 });
	btree->insert({ { { 5, 5, 5 },{ 15, 15, 15 } }, 2 });

	printf("Size = %d", static_cast<int>(btree->size()));
	*dummy = static_cast<int>(btree->size());
}

int main()
{
	box_tree btree;
	box_tree* dtree = nullptr;
	cudaMalloc(&dtree, sizeof(box_tree));
	cudaMemcpy(dtree, &btree, sizeof(box_tree), cudaMemcpyHostToDevice);

	int* dummy = nullptr;
	cudaMalloc(&dummy, sizeof(int));

	treeKernel<<<1, 1>>>(dtree, dummy);
	//treeKernel<<<1, 1>>>(dtree, dummy);

	int output = -1;
	cudaMemcpy(&output, dummy, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Output: %d", output);
}