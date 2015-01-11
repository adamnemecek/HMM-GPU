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

// проверка ошибки OpenCL
/*void checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name
		<< " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}*/

// подсчет процента верно распознанных последовательностей
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



// инциализируем OpenCL
bool initializeOpenCL(cl::Context *& context, std::map<std::string,cl::Kernel*> & kernels, cl::CommandQueue *& queue)
{
	// инициализаци€ платформы
	cl_int err;										// переменна€ с кодом ошибки
	cl::vector< cl::Platform > platformList;		// список платформ			
	cl::Platform::get(&platformList);				//получим список доступных платформ
	checkErr(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");

	// создание контекста
	cl_context_properties cprops[3] = 
		{CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};		// зададим свойства дл€ первой платформы
	context = new cl::Context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err);		// создадим контекст устройства с заданными свойствами
	checkErr(err, "Context::Context()");

	// получение устройств дл€ контекста
	cl::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();			// получение списка устройств дл€ контеста
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	// загрузка исходного кода кернела и компил€ци€
	std::ifstream file("kernels.cl");															// загрузка исходного кода кернела
	checkErr(file.is_open() ? CL_SUCCESS:-1, "kernels.cl");
	std::string prog1(std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));		// запись кода в строку
	file.close();
	cl::Program::Sources sources; //sources(1, std::make_pair(prog1.c_str(), prog1.length()+1));;
	sources.push_back(std::make_pair(prog1.c_str(), prog1.length()+1));				// переменна€ - пара: исходный код + длина кода
	cl::Program program(*context, sources);						// переменна€-программа дл€ данного контекста
	err = program.build(devices,"");							// построение исходного кода дл€ всех устройств
	// выведем ошибки компил€ции, если таковые присутствуют
	std::string buildLog;
	program.getBuildInfo(devices[0],CL_PROGRAM_BUILD_LOG,&buildLog);
	std::cerr << "build log:\n" << buildLog << std::endl;
	std::fstream f;
	f.open("compiler.txt",std::fstream::out);
	f<<buildLog;
	f.close();
	checkErr(err, "Program::build()");
	
	// получение интерфейсов дл€ кернела
	char * kernelsNames[26] = {"calc_g","calcB","k_2_1","k_2_2","k_2_3_1","k_2_3_2",
		"k_2_3_3","k_2_4","k_2_5_1","k_2_5_2","k_2_6","k_2_7","k_3_1_1","k_3_1_2","k_3_2_1",
		"k_3_2_2","k_3_3","k_3_4","k_3_5","k_3_6","k_3_7","k_3_8","k_3_9","k_3_10","k_3_11","k_3_12"};
	for ( int i=0; i<26; i++)
	{
		kernels[kernelsNames[i]] = new cl::Kernel(program, kernelsNames[i],&err);
		checkErr(err, "new cl::Kernel()");
	}

	// инициализаци€ очереди
	queue = new cl::CommandQueue(*context, devices[0], 0, &err);	// очередь команд дл€ 0-го устройства
	checkErr(err, "CommandQueue::CommandQueue()");

	return true;
}

int main(void)
{
	/// таймер
	LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
	/// таймер

	///
	/// »Ќ»÷»јЋ»«ј÷»я OpenCL
	cl::Context * context; std::map<std::string,cl::Kernel*> kernels; cl::CommandQueue * queue;
	initializeOpenCL(context,kernels,queue);
	/// »Ќ»÷»јЋ»«ј÷»я OpenCL
	///


	//
	// ¬џ„»—Ћ≈Ќ»я ѕј–јћ≈“–ќ¬ ћќƒ≈Ћ» “”“ј
	// считаем обучающие последовательности и начальные приближени€ дл€ проведени€ Ѕаума-¬елша
	HMM M1("model1\\"); M1.bindOpenCL(context,kernels,queue);
	HMM M2("model2\\"); M2.bindOpenCL(context,kernels,queue);
    QueryPerformanceCounter(&t1);	// start timer
	M1.findModelParameters();
	M2.findModelParameters();
    QueryPerformanceCounter(&t2);	// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Suck sess\n"); printf("Elapsed time = %f s.\n",elapsedTime);
	// ¬џ„»—Ћ≈Ќ»я ѕј–јћ≈“–ќ¬ ћќƒ≈Ћ» “”“ј
	//
	

	//
	//  Ћј——»‘» ј÷»я
	int K = M1.K;
	cl_float * p1_1 = new cl_float[K]; for(int i=0; i<K; i++) p1_1[i]=0.;
	cl_float * p1_2 = new cl_float[K]; for(int i=0; i<K; i++) p1_2[i]=0.;
	cl_float * p2_1 = new cl_float[K]; for(int i=0; i<K; i++) p2_1[i]=0.;
	cl_float * p2_2 = new cl_float[K]; for(int i=0; i<K; i++) p2_2[i]=0.;
	M1.getTestObserv("model1\\Otest1.txt");		// считаем 1 тест в 1 модель
	M2.getTestObserv("model1\\Otest1.txt");		// считаем 1 тест в 2 модель
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyObservations(p1_1);				// классификаци€ последовательностей 1 типа 1 моделью
	M2.classifyObservations(p1_2);				// классификаци€ последовательностей 1 типа 2 моделью
    QueryPerformanceCounter(&t2);				// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	M1.getTestObserv("model1\\Otest2.txt");		// считаем 2 тест в 1 модель
	M2.getTestObserv("model1\\Otest2.txt");		// считаем 2 тест в 2 модель
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyObservations(p2_1);				// классификаци€ последовательностей 2 типа 1 моделью
	M2.classifyObservations(p2_2);				// классификаци€ последовательностей 2 типа 2 моделью
    QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime += (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Classification complete\nElapsed time = %f s.\n",elapsedTime);
	//  Ћј——»‘» ј÷»я
	//

	/*using namespace std;
	cout << "p1_1\tp1_2" << endl;
	for(int i=0; i<K; i++)
		cout << p1_1[i] << "\t" << p1_2[i] << endl;
	cout << endl << "p2_1\tp2_2" << endl;
	for(int i=0; i<K; i++)
		cout << p2_1[i] << "\t" << p2_2[i] << endl;*/

	/// подсчет процента верно распознанных и запись его в файл
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