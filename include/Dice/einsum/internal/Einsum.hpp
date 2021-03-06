#ifndef HYPERTRIE_EINSUM_HPP
#define HYPERTRIE_EINSUM_HPP

#include <utility>
#include <tsl/hopscotch_set.h>

#include "Dice/einsum/internal/Operator.hpp"
#include "Dice/einsum/internal/CartesianOperator.hpp"
#include "Dice/einsum/internal/JoinOperator.hpp"
#include "Dice/einsum/internal/ResolveOperator.hpp"
#include "Dice/einsum/internal/CountOperator.hpp"
#include "Dice/einsum/internal/EntryGeneratorOperator.hpp"
#include "Dice/einsum/internal/Context.hpp"


struct lookup{
    char label;
    unsigned weight;
    size_t node;
};

namespace einsum::internal {

	template<typename value_type, typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	std::shared_ptr<Operator<value_type, key_part_type, map_type, set_type>>
	Operator<value_type, key_part_type, map_type, set_type>::construct(const std::shared_ptr<Subscript> &subscript,
																	   const std::shared_ptr<Context> &context) {
	    // hier an der stelle wird gehooked d.h
		switch (subscript->type) {
			case Subscript::Type::Join: // std fall
				return std::make_shared<JoinOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																									 context);
			case Subscript::Type::Resolve: // ergebnisse vorkommen sollen
				return std::make_shared<ResolveOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																										context);
			case Subscript::Type::Count: // // lonely auflösen , dh wenn sie nicht vorkommen soll
				return std::make_shared<CountOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																									  context);
			case Subscript::Type::Cartesian: //
				return std::make_shared<CartesianOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																										  context);
			case Subscript::Type::EntryGenerator: //
				return std::make_shared<EntryGeneratorOperator<value_type, key_part_type, map_type, set_type>>(
						subscript, context);
			default:
				throw std::invalid_argument{"subscript is of an undefined type."};
		}
	}

	template<typename value_type, typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	class Einsum {

		using const_BoolHypertrie_t = const_BoolHypertrie<key_part_type, map_type, set_type>;
		using Join_t = Join<key_part_type, map_type, set_type>;
		using Operator_t = Operator<value_type, key_part_type, map_type, set_type>;
		using Entry_t = Entry<key_part_type, value_type>;
		using Key_t = typename Entry_t::key_type;


		std::shared_ptr<Subscript> subscript{};
		std::shared_ptr<Context> context{};
		std::vector<const_BoolHypertrie_t> operands{};
		std::shared_ptr<Operator_t> op{};
		Entry_t entry{};


	public:
		Einsum() = default;

		Einsum(std::shared_ptr<Subscript> subscript, const std::vector<const_BoolHypertrie_t> &operands,
			   TimePoint timeout = TimePoint::max())
				: subscript(std::move(subscript)), context{std::make_shared<Context>(timeout)},
				  operands(operands),
				  op{Operator_t::construct(this->subscript, context)},
				  entry{0, Key_t(this->subscript->resultLabelCount(), std::numeric_limits<key_part_type>::max())} {

		}

		[[nodiscard]] const std::shared_ptr<Subscript> &getSubscript() const {
			return subscript;
		}

		const std::vector<const_BoolHypertrie_t> &getOperands() const {
			return operands;
		}

		const std::shared_ptr<Operator_t> &getOp() const {
			return op;
		}

		struct iterator {
		private:

			std::shared_ptr<Operator_t> op;
			Entry_t *current_entry;
			bool ended_ = false;
		public:
			iterator() = default;

			explicit iterator(Einsum &einsum, Entry_t &entry) : op(einsum.op), current_entry{&entry} {}

			iterator &operator++() {
				op->next();
				return *this;
			}

			inline const Entry<key_part_type, value_type> &operator*() {
				return *current_entry;
			}

			inline const Entry<key_part_type, value_type> &value() {
				return *current_entry;
			}

			operator bool() const {
				return not op->ended();

			}

			[[nodiscard]] inline bool ended() const { return op->ended(); }

		};

		iterator begin() {
			op->load(operands, entry);
			return iterator{*this, entry};
		}

		[[nodiscard]] bool end() const {
			return false;
		}

		void clear() {
		}
	};

	template<typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	class Einsum<bool, key_part_type, map_type, set_type> {
		using value_type = bool;

		using const_BoolHypertrie_t = const_BoolHypertrie<key_part_type, map_type, set_type>;
		using Join_t = Join<key_part_type, map_type, set_type>;
		using Operator_t = Operator<value_type, key_part_type, map_type, set_type>;
		using Entry_t = Entry<key_part_type, bool>;
		using Key_t = typename Entry_t::key_type;

		std::shared_ptr<Subscript> subscript{};
		std::shared_ptr<Context> context{};
		std::vector<const_BoolHypertrie_t> operands{};
		std::shared_ptr<Operator_t> op{};
		Entry_t entry{};

	public:
		Einsum(std::shared_ptr<Subscript> subscript, const std::vector<const_BoolHypertrie_t> &operands,
			   TimePoint timeout = std::numeric_limits<TimePoint>::max())
				: subscript(std::move(subscript)), context{std::make_shared<Context>(timeout)},
				  operands(operands),
				  op{Operator_t::construct(this->subscript, context)},
				  entry{false, Key_t(this->subscript->resultLabelCount(), std::numeric_limits<key_part_type>::max())} {
		    // set Properties von Einsum
		}

		[[nodiscard]] const std::shared_ptr<Subscript> &getSubscript() const {
			return subscript;
		}

		const std::vector<const_BoolHypertrie_t> &getOperands() const {
			return operands;
		}

		const std::shared_ptr<Operator_t> &getOp() const {
			return op;
		}

		struct iterator {
		private:


			std::shared_ptr<Operator_t> op;
			tsl::hopscotch_set<Key_t, ::einsum::internal::KeyHash<key_part_type>> found_entries{};
			Entry_t *current_entry;
			bool ended_ = false;
		public:
			iterator() = default;

			explicit iterator(Einsum &einsum, Entry_t &entry) : op(einsum.op), current_entry{&entry} {
				if (not op->ended()) {
					found_entries.insert(current_entry->key);
				}
			}

			iterator &operator++() {
				op->next();
				while (not op->ended()) {
					assert(current_entry->value == true);
					if (found_entries.find(current_entry->key) == found_entries.end()) {
						found_entries.insert(current_entry->key);
						return *this;
					}
					op->next();
				}
				return *this;
			}

			inline const Entry<key_part_type, value_type> &operator*() {
				return *current_entry;
			}

			inline const Entry<key_part_type, value_type> &value() {
				return *current_entry;
			}

			operator bool() const {
				return not op->ended();

			}

			[[nodiscard]] inline bool ended() const { return op->ended(); }

		};

		iterator begin() {
			op->load(operands, entry);
			return iterator{*this, entry};
		}

		[[nodiscard]] bool end() const {
			return false;
		}

		void clear() {
			throw std::logic_error("not yet implemented.");
		}
	};
}
#endif //HYPERTRIE_EINSUM_HPP
