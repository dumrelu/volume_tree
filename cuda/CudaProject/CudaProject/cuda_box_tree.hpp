#pragma once

#include "bounding_box.hpp"
#include "volume_tree.hpp"
#include "host_allocator.hpp"

#include <vector>

namespace ppc
{
	namespace cuda
	{
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
		using value_type = box_tree::value_type;

		void insert_values(std::vector<value_type> values);
		std::vector<int> find_values(const std::vector<bbox>& bboxes);
		void print_values();
	}
}