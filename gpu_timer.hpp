#pragma once

#include <cstdio>
#include <cuda_runtime.h>

/* Comes from here:
 * https://stackoverflow.com/questions/7876624/timing-cuda-operations
 */
struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	void PrintElapsed()
	{
		float elapsed;

		Stop();
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);

		printf("Elapsed time: %f ms\n", elapsed);
		Start();
	}
};
