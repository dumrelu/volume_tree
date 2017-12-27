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
		static T* allocate()
		{
			return new T{};
		}

		template <typename T>
		static void deallocate(T* ptr)
		{
			delete T;
		}
	};
}