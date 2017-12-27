#pragma once

#include <algorithm>

namespace ppc
{
	template <typename T>
	struct point
	{
		using scalar_type = T;

		scalar_type x;
		scalar_type y;
		scalar_type z;

		template <typename U>
		bool operator==(const point<U>& other) const
		{
			return x == other.x
				&& y == other.y
				&& z == other.z;
		}

		template <typename U>
		bool operator!=(const point<U>& other) const
		{
			return !(*this == other);
		}
	};

	template <typename T>
	struct bounding_box
	{
		using scalar_type = T;
		using point_type = point<scalar_type>;

		point_type min{};
		point_type max{};

		template <typename U>
		bool operator==(const bounding_box<U>& other) const
		{
			return min == other.min && max == other.max;
		}

		template <typename U>
		bool operator!=(const bounding_box<U>& other) const
		{
			return !(*this == other);
		}
	};

	struct Expand
	{
		template <typename T>
		void operator()(bounding_box<T>& box, const typename bounding_box<T>::point_type& point)
		{
			const auto& min = box.min;
			const auto& max = box.max;

			box.min = { std::min(min.x, point.x), std::min(min.y, point.y), std::min(min.z, point.z) };
			box.max = { std::max(max.x, point.x), std::max(max.y, point.y), std::max(max.z, point.z) };
		}

		template <typename T>
		void operator()(bounding_box<T>& box1, const bounding_box<T>& box2)
		{
			(*this)(box1, box2.min);
			(*this)(box1, box2.max);
		}
	};

	struct Intersect
	{
		template <typename T>
		bool operator()(const bounding_box<T>& box1, const bounding_box<T>& box2)
		{
			bool noIntersection =
				box1.max.x < box2.min.x || box1.min.x > box2.max.x ||
				box1.max.y < box2.min.y || box1.min.y > box2.max.y ||
				box1.max.z < box2.min.z || box1.min.z > box2.max.z;
			return !noIntersection;
		}
	};
}