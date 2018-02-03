#pragma once

#include "config.hpp"

namespace ppc
{
	struct host_allocator
	{
		template <typename T>
		struct ptr_type
		{
			using type = T*;
		};

		template <typename T>
		PPC_MEMBER_FUNCTION static T* allocate(T value)
		{
			return new T{ std::move(value) };
		}

		template <typename T>
		PPC_MEMBER_FUNCTION static void deallocate(T* ptr)
		{
			delete ptr;
		}
	};
}