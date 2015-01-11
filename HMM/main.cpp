#include <utility>
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include "HMM.h"
#include <windows.h>                // for Windows APIs

// �������� ������ OpenCL
/*void checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name
		<< " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}*/

// ������� �������� ����� ������������ �������������������
void classClassify(cl_float * p1, cl_float * p2, cl_float &percent1, cl_float &percent2, int K)
{
	percent1=percent2=0;
	for(int k=0;k<K;k++)
		if(p1[k]>p2[k])
			percent1++;
		else
			percent2++;
	percent1/=K;
	percent2/=K;
}



// ������������� OpenCL
bool initializeOpenCL(cl::Context *& context, std::map<std::string,cl::Kernel*> & kernels, cl::CommandQueue *& queue)
{
	// ������������� ���������
	cl_int err;										// ���������� � ����� ������
	cl::vector< cl::Platform > platformList;		// ������ ��������			
	cl::Platform::get(&platformList);				//������� ������ ��������� ��������
	checkErr(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");

	// �������� ���������
	cl_context_properties cprops[3] = 
		{CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};		// ������� �������� ��� ������ ���������
	context = new cl::Context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err);		// �������� �������� ���������� � ��������� ����������
	checkErr(err, "Context::Context()");

	// ��������� ��������� ��� ���������
	cl::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();			// ��������� ������ ��������� ��� ��������
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	// �������� ��������� ���� ������� � ����������
	std::ifstream file("kernels.cl");															// �������� ��������� ���� �������
	checkErr(file.is_open() ? CL_SUCCESS:-1, "kernels.cl");
	std::string prog1(std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));		// ������ ���� � ������
	file.close();
	cl::Program::Sources sources; //sources(1, std::make_pair(prog1.c_str(), prog1.length()+1));;
	sources.push_back(std::make_pair(prog1.c_str(), prog1.length()+1));				// ���������� - ����: �������� ��� + ����� ����
	cl::Program program(*context, sources);						// ����������-��������� ��� ������� ���������
	err = program.build(devices,"");							// ���������� ��������� ���� ��� ���� ���������
	// ������� ������ ����������, ���� ������� ������������
	std::string buildLog;
	program.getBuildInfo(devices[0],CL_PROGRAM_BUILD_LOG,&buildLog);
	std::cerr << "build log:\n" << buildLog << std::endl;
	std::fstream f;
	f.open("compiler.txt",std::fstream::out);
	f<<buildLog;
	f.close();
	checkErr(err, "Program::build()");
	
	// ��������� ����������� ��� �������
	char * kernelsNames[26] = {"calc_g","calcB","k_2_1","k_2_2","k_2_3_1","k_2_3_2",
		"k_2_3_3","k_2_4","k_2_5_1","k_2_5_2","k_2_6","k_2_7","k_3_1_1","k_3_1_2","k_3_2_1",
		"k_3_2_2","k_3_3","k_3_4","k_3_5","k_3_6","k_3_7","k_3_8","k_3_9","k_3_10","k_3_11","k_3_12"};
	for ( int i=0; i<26; i++)
	{
		kernels[kernelsNames[i]] = new cl::Kernel(program, kernelsNames[i],&err);
		checkErr(err, "new cl::Kernel()");
	}

	// ������������� �������
	queue = new cl::CommandQueue(*context, devices[0], 0, &err);	// ������� ������ ��� 0-�� ����������
	checkErr(err, "CommandQueue::CommandQueue()");

	return true;
}

int main(void)
{
	/// ������
	LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
	/// ������

	///
	/// ������������� OpenCL
	cl::Context * context; std::map<std::string,cl::Kernel*> kernels; cl::CommandQueue * queue;
	initializeOpenCL(context,kernels,queue);
	/// ������������� OpenCL
	///


	//
	// ���������� ���������� ������ ����
	// ������� ��������� ������������������ � ��������� ����������� ��� ���������� �����-�����
	HMM M1("model1\\"); M1.bindOpenCL(context,kernels,queue);
	HMM M2("model2\\"); M2.bindOpenCL(context,kernels,queue);
    QueryPerformanceCounter(&t1);	// start timer
	M1.findModelParameters();
	M2.findModelParameters();
    QueryPerformanceCounter(&t2);	// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Suck sess\n"); printf("Elapsed time = %f s.\n",elapsedTime);
	// ���������� ���������� ������ ����
	//
	

	//
	// �������������
	int K = M1.K;
	cl_float * p1_1 = new cl_float[K]; for(int i=0; i<K; i++) p1_1[i]=0.;
	cl_float * p1_2 = new cl_float[K]; for(int i=0; i<K; i++) p1_2[i]=0.;
	cl_float * p2_1 = new cl_float[K]; for(int i=0; i<K; i++) p2_1[i]=0.;
	cl_float * p2_2 = new cl_float[K]; for(int i=0; i<K; i++) p2_2[i]=0.;
	M1.getTestObserv("model1\\Otest1.txt");		// ������� 1 ���� � 1 ������
	M2.getTestObserv("model1\\Otest1.txt");		// ������� 1 ���� � 2 ������
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyObservations(p1_1);				// ������������� ������������������� 1 ���� 1 �������
	M2.classifyObservations(p1_2);				// ������������� ������������������� 1 ���� 2 �������
    QueryPerformanceCounter(&t2);				// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	M1.getTestObserv("model1\\Otest2.txt");		// ������� 2 ���� � 1 ������
	M2.getTestObserv("model1\\Otest2.txt");		// ������� 2 ���� � 2 ������
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyObservations(p2_1);				// ������������� ������������������� 2 ���� 1 �������
	M2.classifyObservations(p2_2);				// ������������� ������������������� 2 ���� 2 �������
    QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime += (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Classification complete\nElapsed time = %f s.\n",elapsedTime);
	// �������������
	//

	/*using namespace std;
	cout << "p1_1\tp1_2" << endl;
	for(int i=0; i<K; i++)
		cout << p1_1[i] << "\t" << p1_2[i] << endl;
	cout << endl << "p2_1\tp2_2" << endl;
	for(int i=0; i<K; i++)
		cout << p2_1[i] << "\t" << p2_2[i] << endl;*/

	/// ������� �������� ����� ������������ � ������ ��� � ����
	cl_float succ1,fail1,succ2,fail2;
	classClassify(p1_1,p1_2,succ1,fail1,K);
	classClassify(p2_1,p2_2,fail2,succ2,K);
	std::fstream f;
	f.open("ClassClassify.txt",std::fstream::out);
	f<<(succ1+succ2)*0.5;
	f.close();

	std::cout << "Percent = " << (succ1+succ2)*0.5 << std::endl;

	return EXIT_SUCCESS;
}