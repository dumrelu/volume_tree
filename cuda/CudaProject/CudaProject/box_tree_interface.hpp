#pragma once

#include "bounding_box.hpp"

#include <utility>
#include <vector>
#include <string>

namespace ppc
{
	namespace box_tree
	{
		using bbox = ppc::bounding_box<int>;
		using bboxes = std::vector<bbox>;
		using value = int;
		using values = std::vector<value>;
		using box_value_pair = std::pair<bbox, value>;
		using box_value_pairs = std::vector<box_value_pair>;

		struct box_tree_interface
		{
			virtual ~box_tree_interface() = default;

			virtual std::string get_name() const = 0;

			virtual void insert(const box_value_pair& value) = 0;
			virtual void insert(const box_value_pairs& values) = 0;
			virtual std::size_t size() const = 0;
			virtual values find(const bboxes& bboxes) const = 0;
		};
	}
}