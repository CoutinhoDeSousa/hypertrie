#ifndef HYPERTRIE_CARDINALITYESTIMATION_HPP
#define HYPERTRIE_CARDINALITYESTIMATION_HPP

#include <cmath>
#include "Dice/einsum/internal/Subscript.hpp"
#include "Dice/einsum/internal/Entry.hpp"
#include "Context.hpp"
extern "C"
{

#include <igraph/igraph.h>
#include <igraph/igraph_cliques.h>  // import for 3.3 igraph_maximal_independent_vertex_sets
}

namespace einsum::internal {


	template<typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	struct CardinalityEstimation {
		using const_BoolHypertrie_t = const_BoolHypertrie<key_part_type, map_type, set_type>;

		/**
		 * Method to get the LAbel with the smallest cardinality (first found if many have the same cardinality)
		 * @param operands
		 * @param label_candidates
		 * @param sc
		 * @return
		 */
		static Label getMinCardLabel(const std::vector<const_BoolHypertrie_t> &operands,
		                             const std::shared_ptr<Subscript> &sc,
		                             std::shared_ptr<Context> context) {
			// Max Idenpended Set gibts schon ? subscribt hash auf Represäntation auf is matched, bau es mir
			// getIndendetSet()
			const tsl::hopscotch_set <Label> &operandsLabelSet = sc->getOperandsLabelSet();
            std::set<Label> labelsss{operandsLabelSet.begin(),operandsLabelSet.end()};
			// lonely labels evtl rausschmeissen, weil die anders übergangen werden
            // wenn context.mis == "none" dann bau ihn hier . sonst nehm den aus context
            // map<Subscript, x> : welches Label wird für dieses Subscript verwendet
            /*
             * Umbauen des mis berechnung:
             * Wenn es in der Map < Subscript, Label > eintrag: dann nimm den
             * sonst guck im MIS nach , nicht vorhandne?
             * dann bau mis neu für dieses Subscript
             * map1 und map2 sind im context
             * map1 "best_label": < Subscript, Label >
             * map2 "label_candidates": < Subscript, Set<Label> > subscript -> MIS (evtl. subset, weil schon welche rausgenommen wurden)
             * if ( subscr in mao1)
             *    // benutze map1[subscr]
             * else if ( subscr in map2 )
             * xxxxxxx:
             *   // l = map2[subscr][0]
             *   // map1[subscr] = map2[subscr][0] //begin unten drunter beginnt ab stelle l
             *   // map2[subscr.remove(l)] = vector{map2[subscr].begin()+1, map2[subscr].end()}
             * else
             *   // mis bauen
             *   // map2[subscr] = mis
             *   // goto: xxxxxxx
             */

            Label returnLabel;
            auto x = context->best_label[*sc];
            if (x==char(0)){ //leer
                if( context->label_candidates.find(*sc) == context->label_candidates.end()){
                    context->label_candidates[*sc] = getMWIS(sc->getRawSubscript().operands);
                }
                auto bestcandidate = context->label_candidates[*sc].begin();
                //context->best_label[*sc] = labelcandidates.begin()->se
                context->best_label[*sc] = *bestcandidate;
                //hier lösch ich vermutlich für alle den weil ich auf dem globalen pointer arbeite?

                if(context->label_candidates[*sc].size() >= 2){
                auto start = std::next(context->label_candidates[*sc].begin(),1);
                context->label_candidates[*sc->removeLabel(*bestcandidate)] =
                            std::set<Label>{start,
                                            context->label_candidates[*sc].end()};
                }
                else if (sc->getRawSubscript().operands.size() == 0){ //sonst pack n leeres rein
                    context->label_candidates[*sc->removeLabel(*bestcandidate)] =
                            std::set<Label>();
                }

            }
            if(context->best_label[*sc] != char(0)){
            returnLabel = context->best_label[*sc];
            }
            else{
                //sonst mach das gleihce wie voerher vorerst
                const tsl::hopscotch_set <Label> &lonely_non_result_labels = sc->getLonelyNonResultLabelSet();

                Label min_label = *operandsLabelSet.begin();
                double min_cardinality = std::numeric_limits<double>::infinity();
                for (const Label label : operandsLabelSet) {
                    if (lonely_non_result_labels.count(label))
                        continue;
                    const double label_cardinality = calcCard(operands, label, sc);
                    if (label_cardinality < min_cardinality) {
                        min_cardinality = label_cardinality;
                        min_label = label;
                    }
                }
                return min_label;
            }


			if (context->label_candidates[*sc].size() == 0){
			//if (operandsLabelSet.size() == 1 || context->label_candidates[*sc].size() == 0){
				return *operandsLabelSet.begin();
			} else {
                return returnLabel;
            }

				// std Min label wird zurück gegeben , wenn alle nicht im result sind klommt das standard min label (das erste )
				//return min_label;
			}


	protected:
		/**
		 * Calculates the cardinality of an Label in an Step.
		 * @tparam T type of the values hold by processed Tensors (Tensor).
		 * @param operands Operands for this Step.
		 * @param step current step
		 * @param label the label
		 * @return label's cardinality in current step.
		 */
		static double calcCard(const std::vector<const_BoolHypertrie_t> &operands, const Label label,
		                       const std::shared_ptr<Subscript> &sc) {
			// get operands that have the label
			const std::vector<LabelPos> &op_poss = sc->getPossOfOperandsWithLabel(label);
			std::vector<double> op_dim_cardinalities(op_poss.size(), 1.0);
			auto label_count = 0;
			auto min_dim_card = std::numeric_limits<size_t>::max();
			tsl::hopscotch_set <size_t> sizes{};

			const LabelPossInOperands &label_poss_in_operands = sc->getLabelPossInOperands(label);
			// iterate the operands that hold the label
			for (auto[i, op_pos] : iter::enumerate(op_poss)) {
				const auto &operand = operands[op_pos];
				const auto op_dim_cards = operand.getCards(label_poss_in_operands[op_pos]);
				const auto min_op_dim_card = *std::min_element(op_dim_cards.cbegin(), op_dim_cards.cend());
				const auto max_op_dim_card = *std::max_element(op_dim_cards.cbegin(), op_dim_cards.cend());

				for (const auto &op_dim_card : op_dim_cards)
					sizes.insert(op_dim_card);

				label_count += op_dim_cards.size();
				// update minimal dimenension cardinality
				if (min_op_dim_card < min_dim_card)
					min_dim_card = min_op_dim_card;

				op_dim_cardinalities[i] = double(max_op_dim_card); //
			}

			std::size_t max_op_size = 0;
			std::vector<std::size_t> op_sizes{};
			for (auto op_pos : op_poss) {
				auto current_op_size = operands[op_pos].size();
				op_sizes.push_back(current_op_size);
				if (current_op_size > max_op_size)
					max_op_size = current_op_size;
			}

			auto const min_dim_card_d = double(min_dim_card);

			double card = std::accumulate(op_dim_cardinalities.cbegin(), op_dim_cardinalities.cend(), double(1),
			                              [&](double a, double b) {
				                              return a * min_dim_card_d / b;
			                              }) / sizes.size();
			return card;
		}
        /**
         * returns for given Operands List the (first) maximum weighted independent set
         * @param operands operand list
         * @return Labelset with the mwis
         */
	    static std::set<Label> getMWIS(const einsum::internal::OperandsSc &operands){

            igraph_t graph;
            igraph_vector_t v;
            igraph_vector_ptr_t mis = IGRAPH_VECTOR_PTR_NULL;
            igraph_vector_ptr_init(&mis, 0);
            /*
             * Start building a graph
             */
            std::vector<int> edges;
            std::map<char,int> edgesList;
            std::map<int,int> weights;
            // zähle 2 maps hoch (label zu id & id zu weight)
            int id = 0;
            for (const auto &operand_sc : operands) {
                for(const auto &op_for_weight: operand_sc){
                    std::pair <std::map<char, int>::iterator, bool> itNodes;
                    itNodes = edgesList.insert( std::pair<char, int>(op_for_weight,id));

                    if(itNodes.second){ //added new
                        weights.insert(std::pair<int,int>(id, 1));
                        id++;
                    }
                    else{ //already in
                        int foundIndex = std::distance(edgesList.begin(), itNodes.first);
                        weights[foundIndex]++;
                    }
                }
                //adding into edgevector based on the operand size

                if(operand_sc.size() == 2){ // just 1 edge, 2 nodes
                    //todo: fehler kontrolle wenn nicht gefunden (füge sie ja oben ein, deshalb nochmal anspechen)
                    //todo: in Methode auslagern
                    edges.push_back(edgesList.find(operand_sc[0])->second);
                    edges.push_back(edgesList.find(operand_sc[1])->second);
                }

                if(operand_sc.size() == 3){ // 3 edges
                    edges.push_back(edgesList.find(operand_sc[0])->second);
                    edges.push_back(edgesList.find(operand_sc[1])->second);
                    edges.push_back(edgesList.find(operand_sc[0])->second);
                    edges.push_back(edgesList.find(operand_sc[2])->second);
                    edges.push_back(edgesList.find(operand_sc[1])->second);
                    edges.push_back(edgesList.find(operand_sc[2])->second);

                }


            }


            int sizeArray = edges.size();
            igraph_real_t e[sizeArray];
            std::copy(edges.begin(),edges.end(),e);
            igraph_vector_view(&v, e, sizeArray);
            igraph_create(&graph,&v, 0 , IGRAPH_UNDIRECTED);

            igraph_maximal_independent_vertex_sets(&graph,&mis);


            // find the "best" mis
            size_t best_mis_id = 0;
            size_t best_size = 0;

            // bestes auswählen anhat der weights (die ich noch nciht habe, aufzaehlen in der for oben
            const size_t set_number_count = igraph_vector_ptr_size(&mis);
            for (size_t i = 0 ; i < set_number_count; i++){
                igraph_vector_t *tmp_mis = (igraph_vector_t*) igraph_vector_ptr_e(&mis,long(i));

                size_t tmp_mis_size = igraph_vector_size(tmp_mis); //const??
                size_t x = 0;
                for (size_t j = 0 ; j < tmp_mis_size; j++){
                    x+= weights[size_t(igraph_vector_e(tmp_mis, long(j)))];
                    if(best_size < x) {
                        best_size = x;
                        best_mis_id = i;
                    }
                }

            }


            igraph_vector_t *best_mis = (igraph_vector_t*)igraph_vector_ptr_e(&mis, long(best_mis_id)); // 0 ersetzen durch "bestes"
            const size_t size_best_mis = igraph_vector_size(best_mis);
            std::vector<size_t> vmis;
            try{
                vmis.resize(size_best_mis);
                for(size_t j = 0 ; j < size_best_mis; j++){
                    vmis[j] += size_t(igraph_vector_e(best_mis, long(j)));
                }
            }
            catch (std::bad_alloc &error) {
                igraph_destroy(&graph);
                igraph_vector_ptr_destroy(&mis); //trz aufräumen und dann weiterschmeissen
                throw error;
            }

            igraph_destroy(&graph);
            igraph_vector_ptr_destroy(&mis);

            /*
             * End building a graph
             */
            // TODO: build it directly into a set instead of a vector that get carried over in a set
            // retransformation to labels

            std::vector<char> vmis_labels;
            std::map<char,int>::const_iterator reverseIt;
            for (auto &[labelx, node_idx] : edgesList){
                auto found = std::find(vmis.begin(), vmis.end(), node_idx);
                if (found != vmis.end())
                    vmis_labels.push_back(labelx);
            }

            return std::set<Label>{vmis_labels.begin(),vmis_labels.end()};
        }
	};
}
#endif //HYPERTRIE_CARDINALITYESTIMATION_HPP
