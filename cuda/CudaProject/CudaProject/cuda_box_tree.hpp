#pragma once

#include "box_tree_interface.hpp"
#include "volume_tree.hpp"
#include "host_allocator.hpp"

namespace ppc
{
	namespace box_tree
	{
		struct cuda_box_tree : public box_tree_interface
		{
			cuda_box_tree();
			~cuda_box_tree();

			virtual std::string get_name() const override
			{
				return "Cuda box tree";
			}

			virtual void insert(const box_value_pair& value) override;
			virtual void insert(const box_value_pairs& values) override;
			virtual std::size_t size() const override;
			virtual values find(const bboxes& bboxe) const override;

		private:
			void check_for_errors() const;
		};
	}
}