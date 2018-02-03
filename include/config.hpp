#pragma once

#ifdef __CUDACC__
#define PPC_MEMBER_FUNCTION	__host__ __device__
#else
#define PPC_MEMBER_FUNCTION	
#endif