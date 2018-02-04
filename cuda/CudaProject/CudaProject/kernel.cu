
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_box_tree.hpp"

#include <stdio.h>
#include <iostream>
#include <chrono>

std::vector<int> find_cpu(const ppc::cuda::box_tree& boxTree, const std::vector<ppc::cuda::bbox>& boxes)
{
	std::vector<int> results;
	for (const auto& box : boxes)
	{
		auto it = boxTree.find(box);
		if (it != boxTree.end())
		{
			results.push_back(it->second);
		}
	}

	return results;
}

int main()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	std::size_t stackSize = 0;
	std::size_t heapSize = 0;
	cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
	cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);

	std::cout << "Stack size: " << stackSize << ", Heap size: " << heapSize << std::endl;

	auto error = cudaDeviceSetLimit(cudaLimitStackSize, 100 * 1024);
	if (error != cudaSuccess)
	{
		std::cerr << "Can't set stack size!" << std::endl;
		return 1;
	}

	/*error = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8388608 * 2);
	if (error != cudaSuccess)
	{
		std::cerr << "Can't set heap size!" << std::endl;
		return 1;
	}*/

	constexpr auto N = 1024;

	ppc::cuda::box_tree btree;
	std::vector<ppc::cuda::value_type> values;
	std::vector<ppc::cuda::bbox> boxes;
	for (auto i = 0; i < N * 2; ++i)
	{
		values.push_back({ { { i*10, i*10, i*10 },{ i*10 + 1, i*10 + 1, i*10 + 1 } }, i });
		btree.insert(values.back());
		boxes.push_back(values.back().first);
	}

	for (int j = 0; j < N; ++j)
	{
		for (auto i = 0; i < N; ++i)
		{
			boxes.push_back(boxes[i]);
		}
	}

	ppc::cuda::insert_values(values);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}
	ppc::cuda::print_values();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}

	{
		const auto start = std::chrono::high_resolution_clock::now();
		auto results = ppc::cuda::find_values(boxes);
		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Num of results: " << results.size() << "Find time: " << duration.count() << " ms" << std::endl;
	}

	{
		const auto start = std::chrono::high_resolution_clock::now();
		auto results = find_cpu(btree, boxes);
		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Num of results: " << results.size() << "Find time: " << duration.count() << " ms" << std::endl;
	}
}

//struct Test
//{
//	int* ptr = nullptr;
//	int x = 0;
//};
//
//__device__ int* g_test = nullptr;
//
//__global__ void newTest(Test* x)
//{
//	/*x->ptr = new int;
//	*x->ptr = 8;*/
//	//cudaMalloc((void**)&x->ptr, sizeof(int));
//
//	if (g_test)
//	{
//		int y = *g_test;
//		printf("g_test = %d\n", y);
//	}
//
//	g_test = (int*)malloc(sizeof(int));
//	*g_test = 8;
//
//	/*x->ptr = (int*) malloc(sizeof(int));
//	*x->ptr = 8;*/
//	//g_test = x->ptr;
//	x->x = *g_test;
//}

/*Test t;
Test* dT = nullptr;
cudaMalloc(&dT, sizeof(Test));
cudaMemcpy(dT, &t, sizeof(Test), cudaMemcpyHostToDevice);

newTest<<<1,1>>>(dT);
cudaDeviceSynchronize();
newTest<<<1,1>>>(dT);

cudaDeviceSynchronize();
cudaMemcpy(&t, dT, sizeof(Test), cudaMemcpyDeviceToHost);
std::cout << t.x << std::endl;

int asd = 0;
cudaMemcpy(&asd, g_test, sizeof(int), cudaMemcpyDeviceToHost);
std::cout << "g_test(host) = " << asd << std::endl;*/


//template <int CudaMemcpyArg>
//struct CudaMemcpy
//{
//	template <typename Ptr>
//	void operator()(Ptr lhs, Ptr rhs)
//	{
//		auto x = sizeof(std::remove_pointer_t<Ptr>);
//		x;
//		cudaMemcpy(lhs, rhs, sizeof(std::remove_pointer_t<Ptr>), static_cast<cudaMemcpyKind>(CudaMemcpyArg));
//	}
//};
//
//using HostToDevice = CudaMemcpy<cudaMemcpyHostToDevice>;
//using DeviceToHost = CudaMemcpy<cudaMemcpyDeviceToHost>;
