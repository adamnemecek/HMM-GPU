#include <utility>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include "HMM.h"
#include <windows.h>

// ������� �������� ����� ������������ �������������������
void classClassify(real_t * p1, real_t * p2, real_t &percent1, real_t &percent2, int K)
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
bool initializeOpenCL(cl::Context *& context, std::map<std::string, cl::Kernel*> & kernels, cl::CommandQueue *& queue)
{
	// ������������� ���������
	cl_int err;										// ���������� � ����� ������
	std::vector< cl::Platform > platformList;		// ������ ��������			
	//cl::Platform::get()
	cl::Platform::get(&platformList);				//������� ������ ��������� ��������
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	std::cout << "Number of platforms: " << platformList.size() << std::endl;

	// ������� ��� ��������� � ����������
	for (int i = 0; i < platformList.size(); i++)
	{
		std::cout << "Platform " << std::to_string(i) << std::endl;
		cl_context_properties cprops[3] =
			{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[i])(), 0 };
		context = new cl::Context(CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &err);	
		std::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
		for (int j = 0; j < devices.size(); j++)
		{
			std::string deviceName;
			devices[j].getInfo(CL_DEVICE_NAME, &deviceName);
			std::cout << deviceName << std::endl;
		}
		delete context;
		std::cout << std::endl;
	}

	// �������� ���������
	// ��������! ����������, �� ������� ����� ����������� ���������� �������� � platformList[�����_����������] !!!!
	cl_context_properties cprops[3] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[1])(), 0 };		// ������� �������� ��� ��������� ���������
	context = new cl::Context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err);	// �������� �������� ���������� � ��������� ����������
	checkErr(err, "Context::Context()");

	// ��������� ��������� ��� ���������
	std::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();			// ��������� ������ ��������� ��� ��������
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");
	// ����������� ���������� �� �����������
	std::string deviceName;
	devices[0].getInfo(CL_DEVICE_NAME, &deviceName);
	std::cout << "Number of devices: " << devices.size() << std::endl;
	std::cout << "Current device: " << deviceName << std::endl;

	// �������� ��������� ���� ������� � ����������
	std::ifstream file("kernels.cl");																// �������� ��������� ���� �������
	checkErr(file.is_open() ? CL_SUCCESS : -1, "kernels.cl");
	std::string prog1(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));		// ������ ���� � ������
	file.close();
	cl::Program::Sources sources; //sources(1, std::make_pair(prog1.c_str(), prog1.length()+1));;
	sources.push_back(std::make_pair(prog1.c_str(), prog1.length() + 1));				// ���������� - ����: �������� ��� + ����� ����
	cl::Program program(*context, sources);											// ����������-��������� ��� ������� ���������
	#if defined(CONFIG_USE_DOUBLE)
		std::cout << "Using double precision" << std::endl;
		err = program.build(devices, "-D CONFIG_USE_DOUBLE");						// ���������� ��������� ���� ��� ���� ��������� ������������ ������� ��������!
	#else
	std::cout << "Using single precision" << std::endl;
		err = program.build(devices, "");
	#endif
	
	// ������� ������ ����������, ���� ������� ������������
	std::string buildLog;
	program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildLog);
	std::cerr << "\nbuild log:\n" << buildLog << std::endl;
	std::fstream f;
	f.open("compiler.txt",std::fstream::out);
	f<<buildLog;
	f.close();
	checkErr(err, "Program::build()");
	
	// ��������� ����������� ��� �������
	char * kernelsNames[41] = {"calc_g","calcB","k_2_1","k_2_2","k_2_3_1","k_2_3_2",
		"k_2_3_3","k_2_4","k_2_5_1","k_2_5_2","k_2_6","k_2_7","k_3_1_1","k_3_1_2","k_3_2_1",
		"k_3_2_2","k_3_3","k_3_4","k_3_5","k_3_6","k_3_7","k_3_8","k_3_9","k_3_10","k_3_11","k_3_12",
		"k_4_0", "k_4_1_1", "k_4_1_2", "k_4_2_1", "k_4_2_2", "k_4_3_1", "k_4_3_2", "k_4_3_3",
		"k_4_4_1", "k_4_4_2", "k_4_4_3", "k_4_4_4", "k_4_5_1", "k_4_5_2", "k_4_5_3"};
	for ( int i = 0; i < 41; i++)
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
	real_t * p1_1 = new real_t[K]; for(int i=0; i<K; i++) p1_1[i]=0.;
	real_t * p1_2 = new real_t[K]; for(int i=0; i<K; i++) p1_2[i]=0.;
	real_t * p2_1 = new real_t[K]; for(int i=0; i<K; i++) p2_1[i]=0.;
	real_t * p2_2 = new real_t[K]; for(int i=0; i<K; i++) p2_2[i]=0.;
	M1.getObservations("model1\\Otest1.txt");		// ������� 1 ���� � 1 ������
	M2.getObservations("model1\\Otest1.txt");		// ������� 1 ���� � 2 ������
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyObservations(p1_1);				// ������������� ������������������� 1 ���� 1 �������
	M2.classifyObservations(p1_2);				// ������������� ������������������� 1 ���� 2 �������
    QueryPerformanceCounter(&t2);				// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	M1.getObservations("model1\\Otest2.txt");		// ������� 2 ���� � 1 ������
	M2.getObservations("model1\\Otest2.txt");		// ������� 2 ���� � 2 ������
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
	real_t succ1,fail1,succ2,fail2;
	classClassify(p1_1,p1_2,succ1,fail1,K);
	classClassify(p2_1,p2_2,fail2,succ2,K);
	std::fstream f;
	f.open("ClassClassify.txt",std::fstream::out);
	f<<(succ1+succ2)*0.5;
	f.close();

	std::cout << "Percent = " << (succ1+succ2)*0.5 << std::endl;


	///
	/// �������� � ������� �����������
	///
	real_t * Olearn1 = new real_t[K * M1.T * M1.Z];
	real_t * Olearn2 = new real_t[K * M2.T * M2.Z];
	HMM * models[2] = { &M1, &M2 };						// ���������� ������ �������
	M1.getObservations("model1\\Ok.txt", Olearn1);		// read learn observations for model 1
	M2.getObservations("model2\\Ok.txt", Olearn2);		// read learn observations for model 2
	real_t * trainingObservations[2] = { Olearn1, Olearn2 };	// ���������� ��������� ���������
	svm_scaling_parameters scalingParameters;
	//QueryPerformanceCounter(&t1);				// start timer
	svm_model * trainedModel = HMM::trainWithDerivatives(trainingObservations, K, models, 2, scalingParameters);
	//QueryPerformanceCounter(&t2);				// stop timer
	//elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	//printf("Derivatives learning complete\nElapsed time = %f s.\n", elapsedTime);
	

	//printf("Derivatives learning complete\n");

	return EXIT_SUCCESS;
}