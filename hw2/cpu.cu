#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>
#include <map>
#include <bitset>
#include <math.h>
#include "error.h"
#include "time.h"
#include "ResizableArray.h"
#include "device_launch_parameters.h"

#define BLOCKNUM   20
#define THREADNUM   1024


struct ItemDetail{
	int id;
	int realId;
	vector<int> tid;
	ItemDetail(int i = -1, int r = -1){
		id = i;
		realId = r;
	}
};

struct Item{
	int id;
	int* db;
	int support;
	Item(int i, int*d, int s){
		id = i;
		db = d;
		support = s;
	}
};

struct EClass{
	vector<Item> items;
	vector<int> parents;
};

const unsigned int Bit32Table[32] =
{
	2147483648UL, 1073741824UL, 536870912UL, 268435456UL,
	134217728, 67108864, 33554432, 16777216,
	8388608, 4194304, 2097152, 1048576,
	524288, 262144, 131072, 65536,
	32768, 16384, 8192, 4096,
	2048, 1024, 512, 256,
	128, 64, 32, 16,
	8, 4, 2, 1
};

const int SIZE_OF_INT = sizeof(int)* 8;
using namespace std;
void ReadInput(FILE *inputFile, int *tNum, int *iNum, int *&index, float supPer, EClass* &root);
void mineGPU(EClass* eClass, int minSup, int* index, int length);
void mineCPU(EClass* eClass, int minSup, int* index, int length);
int NumberOfSetBits(int i);

__global__ static void vecIntersect(int *vec_a, int *vec_b, int *vec_temp, int length, int *support);
__device__  int NumberOfSetBits_dev(int i);
auto out = &cout;
int main(int argc, char** argv){
	clock_t tProgramStart = clock();
	bool cpu = true;
	bool gpu = true;
	char* inFileName = NULL; // the input file name
	float supPer = 0;// user specified minimun support percentage
	if ( argc != 4){//input argument wrong, print usage, return error;
		ErrorHandler(ERROR_INPUT);
	}

	//set arguments
	inFileName = argv[1];
	if ((supPer = atof(argv[2])) == 0 || supPer > 100 || supPer < 0)
		ErrorHandler(ERROR_MINSUP);
	ofstream ofs;
	ofs.open(argv[3], ofstream::out | ofstream::trunc);
	out = &ofs;

	cout << "inFileName = " << inFileName << endl;
	cout << "minSup = " << supPer << endl;

	FILE *inputFile; // input file pointer
	int tNumbers = 0; // Transaction numbers
	int iNumbers = 0; // Item numbers
	int *index = NULL; // the index of item in the database, cause we only want to leave the items that are frequent
	EClass *root = new EClass();
	if ((inputFile = fopen(inFileName, "r")) == 0)
		ErrorHandler(ERROR_INFILE);
	ReadInput(inputFile, &tNumbers, &iNumbers, index, supPer, root);
	int length = tNumbers + SIZE_OF_INT - (tNumbers%SIZE_OF_INT);
	length /= SIZE_OF_INT;
	int minSup = ceil(tNumbers * supPer);
	if (gpu){
		clock_t tGPUMiningStart = clock();
		mineGPU(root, minSup, index, length);
		cout << "Time on GPU Mining: " << (double)(clock() - tGPUMiningStart) / CLOCKS_PER_SEC << endl;
	}
	if (cpu){
		clock_t tCPUMiningStart = clock();
		mineCPU(root, minSup, index, length);
		cout << "Time on CPU Mining: " << (double)(clock() - tCPUMiningStart) / CLOCKS_PER_SEC << endl;
	}
	for (auto item : root->items){
		delete[] item.db;
	}
	delete root;
	delete index;
	return 0;
}

/**
* Read the input from database and store it in memory
* Would filter the items without minimun support
*
* @params
* inputFile: input file pointer
* tNum: record the transaction numbers
* iNum: record the item numbers
* index: conversion from id to real id, used for result output
* supPer: minimun support percentage
* eNum: record the effective item numbers (item with support > minimun support)
*/
void ReadInput(FILE *inputFile, int *tNum, int *iNum, int *&index, float supPer, EClass*&root){
	*tNum = 0;

	map<int, ItemDetail> mapIndex; // store the real id of items and the corresponding ItemDetail.
	char c = 0;
	int temp = 0;
	// read db and convert horizontal database to vertical database and store in the vector of the item in the map
	while ((c = getc(inputFile)) != EOF){
		if (c == ' ' || c == ',' || c == '\n'){
			if (mapIndex.find(temp) == mapIndex.end()){
				mapIndex[temp] = ItemDetail(0, temp);
				mapIndex[temp].tid.push_back(*tNum);
			}
			else mapIndex.find(temp)->second.tid.push_back(*tNum);
			temp = 0;
			if (c == '\n') (*tNum)++;
		}
		else if (47 < c <58){
			temp *= 10;
			temp += c - 48;
		}
	}

	//remove the item without minimun support
	int minSup = (*tNum)*supPer + 1;
	for (map<int, ItemDetail>::iterator it = mapIndex.begin(); it != mapIndex.end();){
		if (it->second.tid.size() < minSup) {
			map<int, ItemDetail>::iterator toErase = it;
			++it;
			// cout<<"hi"<<endl;
			mapIndex.erase(toErase);
		}
		else ++it;
	}

	// convert the tidset into bit vector and store in db, build index
	int bitLength = (*tNum) + SIZE_OF_INT - (*tNum) % SIZE_OF_INT;
//	cout<<"*tNum: "<<*tNum<<" SIZE_OF_INT: "<<SIZE_OF_INT<<" map size: "<<mapIndex.size()<<endl;
	temp = 0;
	index = new int[mapIndex.size()];
	for (map<int, ItemDetail>::iterator it = mapIndex.begin(); it != mapIndex.end(); ++it){
		it->second.id = temp;
		index[temp] = it->second.realId;
		//int * bitVector = (db + temp * bitLength / SIZE_OF_INT);
		int* bitVector = new int[bitLength / SIZE_OF_INT];
		memset(bitVector, 0, sizeof(int)* bitLength / SIZE_OF_INT);
		for (int i = it->second.tid.size() - 1; i >= 0; i--){
			bitVector[it->second.tid[i] / SIZE_OF_INT] |= Bit32Table[it->second.tid[i] % SIZE_OF_INT];
		}
		(*root).items.push_back(Item(temp, bitVector, it->second.tid.size()));
		temp++;
	}
	*iNum = mapIndex.size();
	// for(int i=0;i<*iNum;i++)
		// cout<<index[i]<<endl;
	// cout<<"end"<<endl;
	// cout<<"parents: "<<*((*root).items[2].db)<<endl;	
	// cout<<"id: "<<(*root).items[2].id<<endl;	
}

/**
*	Mining Frequent itemset on GPU
*
*	@Params
*	eClass: pointer to the equivalent class to explore
*	minSup: minimun support
*	index: array that map item id to real id, used for result output
*	length: the length of tidset in integer
*
*/
void mineCPU(EClass *eClass, int minSup, int* index, int length){

	int size = eClass->items.size();

	// cout<<"size is : "<<size<<", length is : "<<length<<endl;
	for (int i = 0; i < size; i++){
		EClass* children = new EClass();
		children->parents = eClass->parents;
		children->parents.push_back(eClass->items[i].id);
		// cout<<"eclass id: "<<eClass->items[i].id<<endl;
		int *a = eClass->items[i].db;
		for (int j = i + 1; j < size; j++){
			int * temp = new int[length];
			int *b = eClass->items[j].db;
			int support = 0;
			for (int k = 0; k < length; k++){
				temp[k] = a[k] & b[k];
				// cout<<"temp[k]: "<<temp[k]<<endl;
				support += NumberOfSetBits(temp[k]);
				// cout<<"support: "<<support<<endl;
			}
			if (support >= minSup){
				children->items.push_back(Item(eClass->items[j].id, temp, support));
			}
			else delete[] temp;
		}
		if (children->items.size() != 0)
			mineCPU(children, minSup, index, length);
		for (auto item : children->items){
			delete[] item.db;
		}
		delete children;
	}
	// output file
	
	for (auto item : eClass->items){
		for (auto i : eClass->parents) *out << index[i] << " ";
		*out << index[item.id] << " (" << item.support << ")" << endl;
	}
}

void mineGPU(EClass *eClass, int minSup, int* index, int length){
	
	// TODO: fill this function to use gpu to accelerate the process of eclat
	int size = eClass->items.size();
	// cout<<"size is : "<<size<<"length is : "<<length<<endl;
	for (int i = 0; i < size; i++){
		EClass* children = new EClass();
		children->parents = eClass->parents;
		children->parents.push_back(eClass->items[i].id);
		
		int *devVec_a,*devVec_temp,*dev_sup,* temp = new int[length], *devVec_b;
		
		cudaMalloc((void**) &devVec_a, sizeof(int) * length);
		cudaMalloc((void**) &dev_sup, sizeof(int) * BLOCKNUM);
		cudaMalloc((void**) &devVec_temp, sizeof(int) * length);
		cudaMalloc((void**) &devVec_b, sizeof(int) * length);

		int *a = eClass->items[i].db;
		cudaMemcpy(devVec_a, a, sizeof(int) * length, cudaMemcpyHostToDevice);
		
		
		for (int j = i + 1; j < size; j++){
			int * temp = new int[length];
			int *support_a = (int *)malloc(sizeof(int)*BLOCKNUM);

			int *b = eClass->items[j].db;
			int support = 0;

			
			cudaMemcpy(devVec_b, b, sizeof(int) * length, cudaMemcpyHostToDevice);

			//doing intersection in device
			vecIntersect<<<BLOCKNUM, THREADNUM, THREADNUM*sizeof(int)*10>>>(devVec_a, devVec_b, devVec_temp,length,dev_sup);
			
			//copy data from device to host
			cudaMemcpy(temp, devVec_temp, sizeof(int) * length, cudaMemcpyDeviceToHost);
			cudaMemcpy(support_a, dev_sup, sizeof(int) * BLOCKNUM, cudaMemcpyDeviceToHost);
			for (int i = 0; i < BLOCKNUM; i++)
			{	
				support += support_a[i];

							
			}			
			free(support_a);
			if (support >= minSup){
				children->items.push_back(Item(eClass->items[j].id, temp, support));
			}
			else delete[] temp;
		}
		cudaFree(devVec_a);
	    cudaFree(dev_sup);
	    cudaFree(devVec_temp);
	    cudaFree(devVec_b);

		if (children->items.size() != 0)
			mineGPU(children, minSup, index, length);
		for (auto item : children->items){
			delete[] item.db;
		}
		delete children;
	}
	// output file
	/*
	for (auto item : eClass->items){
		for (auto i : eClass->parents) *out << index[i] << " ";
		*out << index[item.id] << " (" << item.support << ")" << endl;
	}*/
}


__global__ static void vecIntersect(int *vec_a, int *vec_b, int *vec_temp, int length, int *support)
{

	//tend to do: temp[k] = a[k] & b[k];
	__shared__ int sup_temp[THREADNUM];
    // extern __shared__ int cache_result[];;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int offset = 1, mask = 1;
    int current_tid=threadIdx.x + blockDim.x * blockIdx.x,sup_sum=0;

    // shared[tid] = 0;
    while (current_tid < length) {
	     vec_temp[current_tid]=vec_a[current_tid] & vec_b[current_tid];
		 sup_sum += NumberOfSetBits_dev(vec_temp[current_tid]);
		 current_tid += THREADNUM * BLOCKNUM;  
     }
     // std::cout<<"hi"<<endl;
     
    sup_temp[tid]=sup_sum;
    __syncthreads();

     

    // __syncthreads();
    /*
    if(tid == 0) {
        for(int i = 1; i < THREADNUM; i++) {
            sup_temp[0] += sup_temp[i];
        }
        support[bid] = sup_temp[0];
    }*/
 
    
    while(offset < THREADNUM) {
        if((tid & mask) == 0) {
            sup_temp[tid] += sup_temp[tid + offset];
        }
        offset += offset;
        mask = offset + mask;
        __syncthreads();
    }

    if(tid == 0) {
        support[bid] = sup_temp[0];    

    }


}

int NumberOfSetBits(int i)
{
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
__device__  int NumberOfSetBits_dev(int i)
{
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}


