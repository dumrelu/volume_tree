#pragma once

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
		static T* allocate(T value)
		{
			return new T{ std::move(value) };
		}

		template <typename T>
		static void deallocate(T* ptr)
		{
			delete ptr;
		}
	};
}