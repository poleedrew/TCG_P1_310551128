/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include <fstream>

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), popup(0, 9) {}

	virtual action take_action(const board& after) {
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;
			board::cell tile = popup(engine) ? 1 : 2;
			return action::place(pos, tile);
		}
		return action();
	}

private:
	std::array<int, 16> space;
	std::uniform_int_distribution<int> popup;
};

/**
 * dummy player
 * select a legal action randomly
 */

class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=test role=play " + args),
		opcode({0, 1, 2, 3 }) {action_op = args;}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		if(action_op == "greedy"){
			board::reward value = 0;
			int idx = 0;
			for (int op : opcode) {
				board::reward reward = board(before).slide(op);
				if (reward == -1) continue;
				if( reward >= value){
					value = reward;
					idx = op;
				}
			}
			return action::slide(idx);
		}
		else if(action_op == "tree_search"){
			board::reward value = 0;
			int idx = 0;
			
			for(int op1 : opcode) {
				board origin = board(before);
				board::reward reward1 = origin.slide(op1);
				if (reward1 == -1) continue;
				for( int op2 : opcode){
					board::reward reward2 = origin.slide(op2);
					if (reward2 == -1) continue;
					if(reward1 + reward2 >= value){
						value = reward2;
						idx = op1;
					}
				}
			}
			return action::slide(idx);
		}
		else if(action_op == "heuristic"){
			struct op{
				int code;
				board after;
				int val;
				op(const board& b, int o):code(o), after(b), val(after.slide(code)){}
				bool operator <(const op& b) const{return val < b.val;}
			} op[] = {{before, 0}, {before, 1}, {before, 2}, {before, 3}};
			
			for(int i:{0,1,2,3}){
				if(op[i].val == -1)continue;
				int max_loc = 0;
				int num_space = 0;
				for (int j:{0, 1, 2, 3}){
					board origin = op[i].after;
					if(origin.slide(j) == -1)continue;
					for(int t = 0;t < 16; t++)
						if(origin(t) == 0)
							num_space += 1;
					
					op[i].val += num_space * 5;
				}
				for(int t = 0; t<16;t++){
					if(op[i].after(t) > op[i].after(max_loc))
						max_loc = t;
				}
				if(max_loc == 0 || max_loc == 3 || max_loc == 12 || max_loc == 15)
					op[i].val += op[i].after(max_loc);
				
			}
			std::sort(op, op+4);
			return action::slide(op[3].code);
		}
		else{
			for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
		}
	}	
private:
	std::array<int, 4> opcode;
	std::string action_op;
};