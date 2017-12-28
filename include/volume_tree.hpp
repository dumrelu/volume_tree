#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ppc
{
	namespace detail
	{
		template <
			typename V,
			typename T,
			template <typename> typename Ptr
		>
		struct node
		{
			using volume_type = V;
			using mapped_type = T;
			using value_type = std::pair<volume_type, mapped_type>;
			using value_ptr = Ptr<value_type>;
			using node_ptr = Ptr<node>;

			volume_type volume{};
			value_ptr value{};
			node_ptr parent{};
			node_ptr left{};
			node_ptr right{};

			node(value_ptr val = nullptr)
				: value(val)
			{
				if (value)
				{
					volume = value->first;
				}
			}
		};

		template <
			typename Value, 
			typename NodePtr
		>
		struct iterator
		{
			using value_type = Value;
			using const_value_type = const std::remove_cv_t<value_type>;
			using node_ptr = NodePtr;

			iterator(node_ptr current)
				: m_current{ current }
			{
				if (m_current)
				{
					for (; m_current->left != nullptr; m_current = m_current->left);
				}
			}

			bool operator==(const iterator& other) const
			{
				return m_current == other.m_current;
			}

			bool operator!=(const iterator& other) const
			{
				return !(*this == other);
			}

			value_type& operator*()
			{
				assert(m_current && m_current->value != nullptr);
				return *m_current->value;
			}

			const_value_type& operator*() const
			{
				assert(m_current && m_current->value != nullptr);
				return *m_current->value;
			}

			value_type* operator->()
			{
				assert(m_current && m_current->value != nullptr);
				return &(*m_current->value);
			}

			const_value_type* operator->() const
			{
				assert(m_current && m_current->value != nullptr);
				return &(*m_current->value);
			}

			iterator& operator++()
			{
				assert(m_current != nullptr);

				if (m_current->parent == nullptr)
				{
					m_current = nullptr;
				}
				else if (m_current->parent->left == m_current)
				{
					if (!m_current->parent->right)
					{
						m_current = m_current->parent;
						++(*this);
					}
					else
					{
						m_current = m_current->parent->right;
						for (; m_current->left != nullptr; m_current = m_current->left);
					}
				}
				else //if (m_current->parent->right == m_current)
				{
					m_current = m_current->parent;
					++(*this);
				}

				return *this;
			}

			iterator operator++(int)
			{
				auto tmp = *this;
				++(*this);
				return tmp;
			}

		private:
			node_ptr m_current{};
		};

		template <template <typename> typename Ptr, typename T>
		struct PtrType
		{
			using type = typename Ptr<T>::type;
		};
	}

	template <
		typename V, 
		typename T, 
		typename Intersect, 
		typename Expand, 
		typename Allocator
	>
	class volume_tree
	{
	public:
		using allocator_type = Allocator;
		using volume_intersection = Intersect;
		using volume_expand = Expand;
		template <typename P> using ptr_type = typename detail::PtrType<typename Allocator::ptr_type, P>::type;
		using node_type = detail::node<V, T, ptr_type>;
		using node_ptr = ptr_type<node_type>;
		using volume_type = typename node_type::volume_type;
		using mapped_type = typename node_type::mapped_type;
		using value_type = typename node_type::value_type;
		using iterator = detail::iterator<value_type, node_ptr>;
		using const_iterator = detail::iterator<const value_type, node_ptr>;
		using size_type = std::size_t;
		//TODO: pointers, references, etc

		iterator insert(const value_type& value)
		{
			//TODO: use structured bindings for C++17

			auto valuePtr = m_allocator.allocate<value_type>(value);
			auto newNode = m_allocator.allocate<node_type>(valuePtr);

			if (m_size == 0)
			{
				m_root = newNode;
				++m_size;

				return iterator{ m_root };
			}
			else if (m_size == 1)
			{
				auto newRoot = m_allocator.allocate<node_type>();
				set_left(newRoot, m_root, false);
				set_right(newRoot, newNode);

				m_root = newRoot;
				++m_size;

				return iterator{ m_root->right };
			}
			else
			{
				const auto h = height();

				auto parentPair = find_parent(m_root, 0);
				auto parent = parentPair.first;
				auto parentLevel = static_cast<int>(parentPair.second);

				if (parent && parent->left->value && !parent->right)
				{
					set_right(parent, newNode);
				}
				else
				{
					if (!parent)
					{
						parent = m_allocator.allocate<node_type>();
						parentLevel = -1;
						set_left(parent, m_root, false);
						m_root = parent;
					}

					auto chainPair = allocate_chain(h - 2 - parentLevel);
					auto chainRoot = chainPair.first;
					auto chainEnd = chainPair.second;

					set_right(parent, chainRoot, false);
					set_left(chainEnd, newNode);
				}
				
				++m_size;
				return iterator{ newNode };
			}
		}

		iterator begin() { return iterator{ m_root }; }
		const_iterator begin() const { return const_iterator{ m_root }; }
		const_iterator cbegin() const { return const_iterator{ m_root }; }

		iterator end() { return iterator{ nullptr }; }
		const_iterator end() const { return const_iterator{ nullptr }; }
		const_iterator cend() { return const_iterator{ nullptr }; }

		size_type size() const { return m_size; }

	private:

		size_type height() const
		{
			//TODO: determine without iterating it
			size_type height = 0;
			for (auto node = m_root; node != nullptr; node = node->left, ++height);
			return height;
		}

		std::pair<node_ptr, size_type> find_parent(node_ptr node, size_type level)
		{
			if (node->right)		//Parent node
			{
				return find_parent(node->right, level + 1);
			}
			else if (node->left)	//Parent node
			{
				auto res = find_parent(node->left, level + 1);
				if (!res.first)
				{
					return { node, level };
				}
				return res;
			}
			else if (node->parent->right == node)	//Right child node
			{
				return { nullptr, 0 };
			}
			else	//Left child
			{
				return { node->parent, level - 1 };
			}
		}

		void set_left(node_ptr parent, node_ptr left, bool update = true)
		{
			assert(left);

			parent->left = left;
			left->parent = parent;
			if (update)
			{
				update_volumes(parent);
			}
		}

		void set_right(node_ptr parent, node_ptr right, bool update = true)
		{
			assert(right);

			parent->right = right;
			right->parent = parent;
			if (update)
			{
				update_volumes(parent);
			}
		}

		void update_volumes(node_ptr node)
		{
			while (node)
			{
				update_volume(node);
				node = node->parent;
			}
		}

		void update_volume(node_ptr node)
		{
			assert(node);

			node->volume = {};
			if (node->left)
			{
				m_expand(node->volume, node->left->volume);
			}

			if (node->right)
			{
				m_expand(node->volume, node->right->volume);
			}
		}

		std::pair<node_ptr, node_ptr> allocate_chain(size_type size)
		{
			auto chainRoot = m_allocator.allocate<node_type>();
			auto node = chainRoot;

			while (--size)
			{
				set_left(node, m_allocator.allocate<node_type>(), false);
				node = node->left;
			}
			return { chainRoot, node };
		}

		node_ptr m_root;
		allocator_type m_allocator;
		volume_intersection m_intersects;
		volume_expand m_expand;
		size_type m_size{};
	};
}