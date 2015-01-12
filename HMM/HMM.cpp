#include "HMM.h"
#include "fstream"

void checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name
		<< " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

HMM::HMM(std::string filename)
{
	std::fstream f;
	f.open(filename+"Params.txt",std::fstream::in);
	f>>N>>M>>Z>>T>>K>>NumInit;
	f.close();

	//¬џƒ≈Ћя≈ћ ѕјћя“№ ƒЋя ѕј–јћ≈“–ќ¬ —ћћ
	PI = new real_t[N];				// начальное распределение веро€тностей
	A = new real_t[N*N];				// веро€тности переходов
	TAU = new real_t[N*M];			
	MU = new real_t[N*M*Z];
	SIG = new real_t[N*M*Z*Z];
	alf = new real_t[T*N*K];
	bet = new real_t[T*N*K];
	c = new real_t[T*K];				// коэффициенты масштаба
	ksi = new real_t[(T-1)*N*N*K];	
	gam = new real_t[T*N*K];
	gamd = new real_t[T*N*M*K];
	alf_t = new real_t[T*N*K];
	bet_t = new real_t[T*N*K];
	B = new real_t[N*T*K];			// веро€тности по€влени€ наблюдений

	//начальные приближени€
	A1 = new real_t[N*N];
	TAU1 = new real_t[N*M];
	MU1 = new real_t[N*M*Z*NumInit];
	SIG1 = new real_t[N*M*Z*Z*NumInit];
	PI1 = new real_t[N];
	Otr = new real_t[T*Z*K];

	f.open(filename+"PI1.txt",std::fstream::in);
	for(cl_int i=0;i<N;i++)
		f>>PI1[i];
	f.close();

	f.open(filename+"A1.txt",std::fstream::in);
	for(cl_int i=0;i<N;i++)
		for (cl_int j=0;j<N;j++)
			f>>A1(i,j);
	f.close();

	f.open(filename+"TAU1.txt",std::fstream::in);
	for(cl_int i=0;i<N;i++)
		for (cl_int j=0;j<M;j++)
			f>>TAU1(i,j);
	f.close();

	f.open(filename+"Ok.txt",std::fstream::in);
	for(cl_int k=0;k<K;++k)
		for(cl_int t=0;t<T;t++)
			for (cl_int z=0;z<Z;z++)
				f>>Otr(k,t,z);
	f.close();

	f.open(filename+"MU1.txt",std::fstream::in);
	for (cl_int n=0;n<NumInit;n++)
		for (cl_int i=0;i<N;i++)
			for(cl_int m=0;m<M;m++)
				for(cl_int z=0;z<Z;z++)
					f>>MU1(z,i,m,n);
	f.close();

	f.open(filename+"SIG1.txt",std::fstream::in);
	for (cl_int n=0;n<NumInit;n++)
		for (cl_int i=0;i<N;i++)
			for(cl_int m=0;m<M;m++)
				for(cl_int z1=0;z1<Z;z1++)
					for(cl_int z2=0;z2<Z;z2++)
						f>>SIG1(z1,z2,i,m,n);
	f.close();

	// инициализируем среду OpenCL
	//initializeOpenCL();
}


HMM::~HMM(void)
{
	delete A1;
	delete TAU1;
	delete MU1;
	delete SIG1;
	delete PI1;
	delete Otr; 
}

void HMM::bindOpenCL(cl::Context * context_, std::map<std::string,cl::Kernel*> & kernels_, cl::CommandQueue * queue_) // прив€зка OpenCL переменных и создание буферов
{
	context = context_; kernels = kernels_; queue = queue_;
	// буферы
	// параметры
	PI_b = new cl::Buffer(*context,NULL,N*sizeof(real_t));
	A_b = new cl::Buffer(*context,NULL,N*N*sizeof(real_t));
	TAU_b = new cl::Buffer(*context,NULL,N*M*sizeof(real_t));
	MU_b = new cl::Buffer(*context,NULL,Z*N*M*sizeof(real_t));
	SIG_b = new cl::Buffer(*context,NULL,Z*Z*N*M*sizeof(real_t));
	// начальные приближени€
	PI1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*sizeof(real_t),PI1);
	A1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*N*sizeof(real_t),A1);
	TAU1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*M*sizeof(real_t),TAU1);
	MU1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*M*Z*NumInit*sizeof(real_t),MU1);
	SIG1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*M*Z*Z*NumInit*sizeof(real_t),SIG1);
	// наблюдени€
	Otr_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,K*T*Z*sizeof(real_t),Otr);
	// вспомогательные массивы
	alf_b = new cl::Buffer(*context,NULL,T*N*K*sizeof(real_t));
	bet_b = new cl::Buffer(*context,NULL,T*N*K*sizeof(real_t));
	c_b = new cl::Buffer(*context,NULL,T*K*sizeof(real_t));
	ksi_b = new cl::Buffer(*context,NULL,T*N*N*K*sizeof(real_t));
	gam_b = new cl::Buffer(*context,NULL,T*N*K*sizeof(real_t));
	gamd_b = new cl::Buffer(*context,NULL,T*N*M*K*sizeof(real_t));
	alf_t_b = new cl::Buffer(*context,NULL,T*N*K*sizeof(real_t));
	bet_t_b = new cl::Buffer(*context,NULL,T*N*K*sizeof(real_t));
	B_b = new cl::Buffer(*context,NULL,N*T*K*sizeof(real_t));
	// временный массив
	gam_sum_b = new cl::Buffer(*context,NULL,N*sizeof(real_t));
	gamd_sum_b = new cl::Buffer(*context,NULL,M*N*sizeof(real_t));
	// флаг ошибки
	flag_b = new cl::Buffer(*context,NULL,sizeof(cl_int));
	// g
	g_b = new cl::Buffer(*context,NULL,T*K*N*M*NumInit*sizeof(real_t));
}

/*void HMM::showInfo()
{
	// узнаем об устройствах и кернеле
	cl_uint maxComputeUnits;			// число вычислительных единиц
	size_t maxWorkGroupSize;			// максимальный размер рабочей группы
	size_t prefWorkGroupSizeMul;		// размер wavefront'a
	cl_ulong localMemSize;
	cl_ulong globalMemSize;
	devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,&maxComputeUnits);
	devices[0].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,&maxWorkGroupSize);
	//size_t maxWorkItemSizes[3];
	//devices[0].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES,&maxWorkItemSizes);
	//std::cerr << "maxWorkItemSizes=" << maxWorkItemSizes[0] << " " << maxWorkItemSizes[1] << " " << maxWorkItemSizes[2] << std::endl;
	devices[0].getInfo(CL_DEVICE_LOCAL_MEM_SIZE,&localMemSize);
	std::cerr << "localMemSize = " << localMemSize << " mb" << std::endl;
	devices[0].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE,&globalMemSize);
	std::cerr << "globalMemSize = " << globalMemSize/1024/1024 << " mb" << std::endl;
	size_t maxParameterSize;
	devices[0].getInfo(CL_DEVICE_MAX_PARAMETER_SIZE,&maxParameterSize);
	std::cerr << "maxParameterSize = " << maxParameterSize << " bytes" << std::endl;
	// расширени€
	//std::string extensionsList;
	//devices[i].getInfo(CL_DEVICE_EXTENSIONS,&extensionsList);		//TODO: включать real_t, если такое расширение доступно
	kernel->getWorkGroupInfo(devices[0],CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,&prefWorkGroupSizeMul);	
}*/

// вспомогательна€ функци€
void copyArray(real_t * dest, real_t * source, cl_int n)
{
	for(cl_int i=0; i<n; i++)
		dest[i]=source[i];
}

void HMM::findModelParameters()
{
	cl_int err;
	// выполним Ѕаума ¬елша дл€ всех начальных приближений и выбираем лучший набор параметров
	real_t p, p0 = -1000000000000000.;
	// n - номер приближени€
	for(cl_int n=0; n<NumInit; n++)
	{
		p = calcBaumWelсh(n);
		if(p>p0)
		{
			// TODO: load pi,a,tau,mu,sig from gpu and save here
			cl_int err = queue->enqueueReadBuffer(*PI_b, CL_TRUE, 0, N*sizeof(real_t), PI);	
			checkErr(err, "enqueueReadBuffer() - PI_b");
			err = queue->enqueueReadBuffer(*A_b, CL_TRUE, 0, N*N*sizeof(real_t), A);	
			checkErr(err, "enqueueReadBuffer() - A_b");
			err = queue->enqueueReadBuffer(*TAU_b, CL_TRUE, 0, N*M*sizeof(real_t), TAU);	
			checkErr(err, "enqueueReadBuffer() - TAU_b");
			err = queue->enqueueReadBuffer(*MU_b, CL_TRUE, 0, Z*N*M*sizeof(real_t), MU);	
			checkErr(err, "enqueueReadBuffer() - MU_b");
			err = queue->enqueueReadBuffer(*SIG_b, CL_TRUE, 0, Z*Z*N*M*sizeof(real_t), SIG);	
			checkErr(err, "enqueueReadBuffer() - SIG_b");
			p0=p;
		}
	}

	// распечатать
	/*using namespace std;
	cout << "p=" << p << endl;
	cout << "pi:" << endl;
	for (cl_int i=0; i<N; i++)
		cout << PI[i] << endl;
	cout << "A:" << endl;
	for (cl_int i=0; i<N; i++){
		for (cl_int j=0; j<N; j++)
			cout << A(i,j) << "\t";
		cout << "\n";
	}
	cout << "TAU:" << endl;
	for (cl_int i=0; i<N; i++){
		for (cl_int m=0; m<M; m++)
			cout << TAU(i,m) << "\t";
		cout << "\n";
	}*/

	// back to gpu
	err = queue->enqueueWriteBuffer(*PI_b, CL_TRUE, 0, N*sizeof(real_t), PI);	
	checkErr(err, "enqueueReadBuffer() - PI_b");
	err = queue->enqueueWriteBuffer(*A_b, CL_TRUE, 0, N*N*sizeof(real_t), A);	
	checkErr(err, "enqueueReadBuffer() - A_b");
	err = queue->enqueueWriteBuffer(*TAU_b, CL_TRUE, 0, N*M*sizeof(real_t), TAU);	
	checkErr(err, "enqueueReadBuffer() - TAU_b");
	err = queue->enqueueWriteBuffer(*MU_b, CL_TRUE, 0, Z*N*M*sizeof(real_t), MU);	
	checkErr(err, "enqueueReadBuffer() - MU_b");
	err = queue->enqueueWriteBuffer(*SIG_b, CL_TRUE, 0, Z*Z*N*M*sizeof(real_t), SIG);	
	checkErr(err, "enqueueReadBuffer() - SIG_b");
}

void HMM::classifyObservations(real_t * p)
{
	// внутренние вычислени€
	internal_calculations(-1);

	// get c from GPU
	cl_int err = queue->enqueueReadBuffer(*c_b, CL_TRUE, 0, T*K*sizeof(real_t), c);	
	checkErr(err, "enqueueReadBuffer() - c_b");

	// кернел 5  (параллел)
	/*std::cout << "classification probabilities:" << std::endl;
	for(cl_int k=0;k<K;k++){
		for(cl_int t=0;t<T;t++)
			p[k]-=log(c(t,k));
		std::cout << p[k] << std::endl;
	}*/
	for(cl_int k=0;k<K;k++)
		for(cl_int t=0;t<T;t++)
			p[k]-=log(c(t,k));
}

/*real_t HMM::g(cl_int t,cl_int k,cl_int i,cl_int m,cl_int n)
{
	//работаем с диагональными ковариационными матрицами
	real_t det=1.,res=0.;
	real_t tmp1,tmp2;
	if (n==-1) //работа с уже полученными параметрами модели 
	{
		for (cl_int z=0;z<Z;z++)
		{
			tmp1=SIG(z,z,i,m);
			det*=tmp1;
			tmp2=Otr(k,t,z)-MU(z,i,m);
			res+=tmp2*tmp2/tmp1;
		}		
	}
	else
	{
		for (cl_int z=0;z<Z;z++)
		{
			tmp1=SIG1(z,z,i,m,n);
			det*=tmp1;
			tmp2=Otr(k,t,z)-MU1(z,i,m,n);
			res+=tmp2*tmp2/tmp1;
		}
	}
	res*=-0.5;
	res= (real_t) exp(res)/sqrt((real_t)pow(2.*pi,Z)*det);
	return res;

}*/

real_t HMM::calcBaumWelсh(cl_int n)
{
	cl_int err;
	cl_int T1=T-1;
	cl::Event last_event;

	/*real_t * gam_sum = new real_t[N];
	real_t * gamd_sum = new real_t[N*M];
	real_t * tmp3 = new real_t[Z];
	real_t tmp2;*/
	//vector<double> tmp3(Z);

	for(cl_int iter=0;iter<5;iter++)
	{
		// большой блок вспомогательных вычислений
		internal_calculations(n);

		// кернел 3.1.1
		cl::Kernel * kernel = kernels["k_3_1_1"];
		kernel->setArg(0,*gam_sum_b); 
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
		checkErr(err, "k_3_1_1");

		// кернел 3.1.2
		kernel = kernels["k_3_1_2"];
		kernel->setArg(0,M); kernel->setArg(1,*gamd_sum_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, M), cl::NullRange);
		checkErr(err, "k_3_1_2");
		
		
		// кернел 3.2 (ЅќЋ№Ўќ…)
		cl::Kernel * kernel1 = kernels["k_3_2_1"];
		kernel1->setArg(2,N); kernel1->setArg(3,K);
		kernel1->setArg(4,*gam_sum_b); kernel1->setArg(5,*gam_b);
		kernel1->setArg(6,*flag_b);
		cl::Kernel * kernel2 = kernels["k_3_2_2"];
		kernel2->setArg(2,N); kernel2->setArg(3,M); 
		kernel2->setArg(4,K); kernel2->setArg(5,*gam_sum_b);
		kernel2->setArg(6,*gamd_sum_b); kernel2->setArg(7,*gamd_b);
		kernel2->setArg(8,*flag_b);
		cl_int flag = 0;
		/// FLAG = 0 -> to GPU
		//err = queue->enqueueWriteBuffer(*flag_b, CL_TRUE, 0, 1*sizeof(cl_int), &flag);
		checkErr(err, "enqueueWriteBuffer() - flag_b");
		for(cl_int t=0; t<T1 && flag == 0; t++)
			for(cl_int k=0; k<K && flag == 0; k++)
			{
				// кернел 3.2.1
				kernel1->setArg(0,t); 
				kernel1->setArg(1,k);
				err = queue->enqueueNDRangeKernel(*kernel1, cl::NullRange, cl::NDRange(N), cl::NullRange);
				checkErr(err, "k_3_2_1");
				// кернел 3.2.2
				kernel2->setArg(0,t); 
				kernel2->setArg(1,k);
				err = queue->enqueueNDRangeKernel(*kernel2, cl::NullRange, cl::NDRange(N, M), cl::NullRange);
				checkErr(err, "k_3_2_2");
				// LOAD F from GPU here
				//err = queue->enqueueReadBuffer(*flag_b, CL_TRUE, 0, 1*sizeof(cl_int), &flag);	
				//checkErr(err, "enqueueReadBuffer() - flag_b");
			}

		// check F
		//if(flag < 0) 
			//break;
		
		// кернел 3.3
		kernel = kernels["k_3_3"];
		kernel->setArg(0,N); kernel->setArg(1,K); kernel->setArg(2,*PI_b); kernel->setArg(3,*gam_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
		checkErr(err, "k_3_3");
		
		// кернел 3.4
		kernel = kernels["k_3_4"];
		kernel->setArg(0,K); kernel->setArg(1,N); kernel->setArg(2,T1);
		kernel->setArg(3,*A_b); kernel->setArg(4,*ksi_b);
		kernel->setArg(5,*gam_sum_b); kernel->setArg(6,*c_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, N), cl::NullRange);
		checkErr(err, "k_3_4");

		// кернел 3.5
		kernel = kernels["k_3_5"];
		kernel->setArg(0,M); kernel->setArg(1,*TAU_b); kernel->setArg(2,*gamd_sum_b);
		kernel->setArg(3,*gam_sum_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, M), cl::NullRange);
		checkErr(err, "k_3_5");
		
		// кернел 3.6
		kernel = kernels["k_3_6"];
		kernel->setArg(0,N); kernel->setArg(1,M); kernel->setArg(2,K);
		kernel->setArg(3,Z); kernel->setArg(4,T); 
		kernel->setArg(5,*MU_b); kernel->setArg(6,*gamd_b); 
		kernel->setArg(7,*Otr_b); kernel->setArg(8,*gamd_sum_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, Z, M), cl::NullRange);
		checkErr(err, "k_3_6");

		// кернел 3.7
		kernel = kernels["k_3_7"];
		kernel->setArg(0,N); kernel->setArg(1,M); kernel->setArg(2,Z);
		kernel->setArg(3,K); kernel->setArg(4,T);
		kernel->setArg(5,*SIG_b); kernel->setArg(6,*gamd_b); 
		kernel->setArg(7,*gamd_sum_b); 
		kernel->setArg(8,*MU_b); kernel->setArg(9,*Otr_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, Z*Z, M), cl::NullRange);
		checkErr(err, "k_3_7");
		
		// кернел 3.8
		kernel = kernels["k_3_8"];
		kernel->setArg(0,*PI1_b); kernel->setArg(1,*PI_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
		checkErr(err, "k_3_8");
		// кернел 3.9
		kernel = kernels["k_3_9"];
		kernel->setArg(0,N); kernel->setArg(1,*A1_b); kernel->setArg(2,*A_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, N), cl::NullRange);
		checkErr(err, "k_3_9");
		// кернел 3.10
		kernel = kernels["k_3_10"];
		kernel->setArg(0,M); kernel->setArg(1,*TAU1_b); kernel->setArg(2,*TAU_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, M), cl::NullRange);
		checkErr(err, "k_3_10");
		// кернел 3.11
		kernel = kernels["k_3_11"];
		kernel->setArg(0,N); kernel->setArg(1,M); kernel->setArg(2,Z);
		kernel->setArg(3,n); kernel->setArg(4,*MU1_b); kernel->setArg(5,*MU_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, M, Z), cl::NullRange);
		checkErr(err, "k_3_11");
		// кернел 3.12
		kernel = kernels["k_3_12"];
		kernel->setArg(0,N); kernel->setArg(1,M); kernel->setArg(2,Z);
		kernel->setArg(3,n); kernel->setArg(4,*SIG1_b); kernel->setArg(5,*SIG_b);
		
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N,M,Z*Z),cl::NullRange,NULL,&last_event);
		checkErr(err, "k_3_12");	
	}
	
	// load c here from GPU
	err = last_event.wait();

	return calcProbability();
}


void HMM::internal_calculations(cl_int n)
{
	cl::Event event;
	cl_int err;
	cl_int T1=T-1;
	real_t * TAU_used, * A_used, * PI_used, * SIG_used, * MU_used;
	cl::Buffer * TAU_used_b, * A_used_b, * PI_used_b, *SIG_used_b, * MU_used_b;
	if (n==-1){
		TAU_used = TAU; TAU_used_b = TAU_b;
		A_used = A; A_used_b = A_b;
		PI_used = PI; PI_used_b = PI_b;
		SIG_used = SIG; SIG_used_b = SIG_b;
		MU_used = MU;	MU_used_b = MU_b;	
	}
	else{
		TAU_used = TAU1; TAU_used_b = TAU1_b;
		A_used = A1; A_used_b = A1_b;
		PI_used = PI1; PI_used_b = PI1_b;
		SIG_used = SIG1; SIG_used_b = SIG1_b;
		MU_used = MU1;	MU_used_b = MU1_b;	
	}

	// кернел calc_G
	cl::Kernel * kernel = kernels["calc_g"];
	kernel->setArg(0,n); kernel->setArg(1,N);
	kernel->setArg(2,M); kernel->setArg(3,Z);
	kernel->setArg(4,T); kernel->setArg(5,K);
	kernel->setArg(6,*SIG_used_b); kernel->setArg(7,*Otr_b);
	kernel->setArg(8,*MU_used_b); kernel->setArg(9,*g_b);
	err = queue->enqueueNDRangeKernel(
		*kernel, 
		cl::NullRange,
		cl::NDRange(T, K, N*M),
		cl::NullRange);
	checkErr(err, "calc_g");

	// кернел calcB
	kernel = kernels["calcB"];
	kernel->setArg(0,n); kernel->setArg(1,N);
	kernel->setArg(2,M); kernel->setArg(3,Z);
	kernel->setArg(4,T); kernel->setArg(5,K);
	kernel->setArg(6,*TAU_used_b); kernel->setArg(7,*B_b);
	kernel->setArg(8,*SIG_used_b); kernel->setArg(9,*Otr_b);
	kernel->setArg(10,*MU_used_b); kernel->setArg(11,*g_b);
	err = queue->enqueueNDRangeKernel(
		*kernel, 
		cl::NullRange,
		cl::NDRange(N,T,K),
		cl::NullRange);
	checkErr(err, "calcB");

	// DEBUG - EVERYTHING IS OK!
	/*real_t * B_dbg = new real_t[N*T*K];
	err = queue->enqueueReadBuffer(*B_b, CL_TRUE, 0, N*T*K*sizeof(real_t), B_dbg);	
	checkErr(err, "enqueueReadBuffer() - B_b");
	std::fstream f;
	f.open("debugging_B.txt",std::fstream::out);
	for (cl_int i=0; i<N*T*K; i++)
		f << B_dbg[i] << std::endl;
	f.close();*/
	// /DEBUG - EVERYTHING IS OK!

	// кернел 2.1
	kernel = kernels["k_2_1"];
	kernel->setArg(0,N); kernel->setArg(1,K);
	kernel->setArg(2,*c_b); 
	kernel->setArg(3,*alf_b); kernel->setArg(4,*bet_b); 
	kernel->setArg(5,*alf_t_b); kernel->setArg(6,*bet_t_b);
	err = queue->enqueueNDRangeKernel(
		*kernel, 
		cl::NullRange,
		cl::NDRange(N,T,K),
		cl::NullRange);
	checkErr(err, "k_2_1");



	// кернел 2.2 (set_var)
	kernel = kernels["k_2_2"];
	kernel->setArg(0,N); kernel->setArg(1,K);
	kernel->setArg(2,T1); kernel->setArg(3,*alf_t_b);
	kernel->setArg(4,*PI_used_b); kernel->setArg(5,*B_b);
	kernel->setArg(6,*bet_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
	checkErr(err, "k_2_2");

	// кернел 2.3 (большой)
	cl::Kernel * kernel1 = kernels["k_2_3_1"];
	kernel1->setArg(0,N); kernel1->setArg(1,K);
	kernel1->setArg(3,*alf_b); kernel1->setArg(4,*alf_t_b);
	cl::Kernel * kernel2 = kernels["k_2_3_2"];
	kernel2->setArg(0,N); kernel2->setArg(1,K);
	kernel2->setArg(2,T); kernel2->setArg(4,*alf_b); 
	kernel2->setArg(5,*alf_t_b); kernel2->setArg(6,*A_used_b);
	kernel2->setArg(7,*B_b);
	for(cl_int t=0; t<T1; t++)
	{
		// кернел 2.3.1
		kernel1->setArg(2,t);
		err = queue->enqueueNDRangeKernel(*kernel1, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
		checkErr(err, "k_2_3_1");
		// кернел 2.3.2
		kernel2->setArg(3,t);
		err = queue->enqueueNDRangeKernel(*kernel2, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
		checkErr(err, "k_2_3_2");
	}

	// DEBUG alf, alf_t
	/*real_t * alf_dbg = new real_t[N*T*K];
	err = queue->enqueueReadBuffer(*alf_b, CL_TRUE, 0, N*T*K*sizeof(real_t), alf_dbg);	
	checkErr(err, "enqueueReadBuffer() - alf_b");
	real_t * alf_t_dbg = new real_t[N*T*K];
	err = queue->enqueueReadBuffer(*alf_t_b, CL_TRUE, 0, N*T*K*sizeof(real_t), alf_t_dbg);
	checkErr(err, "enqueueReadBuffer() - alf_t_b");
	f.open("debugging_alf.txt",std::fstream::out);
	for (cl_int i=0; i<N*T*K; i++)
		f << alf_dbg[i] << std::endl;
	f.close();
	f.open("debugging_alf_t.txt",std::fstream::out);
	for (cl_int i=0; i<N*T*K; i++)
		f << alf_t_dbg[i] << std::endl;
	f.close();*/
	// /DEBUG

	// кернел 2.3.3
	kernel = kernels["k_2_3_3"];
	kernel->setArg(0,N); kernel->setArg(1,K);
	kernel->setArg(2,*c_b); kernel->setArg(3,*alf_b);
	kernel->setArg(4,*alf_t_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(T1, K), cl::NullRange);
	checkErr(err, "k_2_3_3");

	


	// кернел 2.4 (set_var)
	kernel = kernels["k_2_4"];
	kernel->setArg(0,N); kernel->setArg(1,K);
	kernel->setArg(2,T1); kernel->setArg(3,*alf_b);
	kernel->setArg(4,*alf_t_b); kernel->setArg(5,*c_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
	checkErr(err, "k_2_4");
	

	// кернел 2.5 (большой)
	kernel1 = kernels["k_2_5_1"];
	kernel1->setArg(1,N);
	kernel1->setArg(2,K); kernel1->setArg(3,*bet_t_b);
	kernel1->setArg(4,*c_b); kernel1->setArg(5,*bet_b);
	kernel2 = kernels["k_2_5_2"];
	kernel2->setArg(1,N); kernel2->setArg(2,K);
	kernel2->setArg(3,T); kernel2->setArg(4,*bet_b);
	kernel2->setArg(5,*A_used_b); kernel2->setArg(6,*B_b);
	kernel2->setArg(7,*bet_t_b); kernel2->setArg(8,*alf_b);
	for(cl_int t=T1-1;t>=0;t--)
	{
		// кернел 2.5.1
		kernel1->setArg(0,t);
		err = queue->enqueueNDRangeKernel(*kernel1, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
		checkErr(err, "k_2_5_1");
		// кернел 2.5.2
		kernel2->setArg(0,t);	
		err = queue->enqueueNDRangeKernel(*kernel2, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
		checkErr(err, "k_2_5_2");
	}

		
	// кернел 2.6 
	kernel = kernels["k_2_6"];
	kernel->setArg(0,n); kernel->setArg(1,N);
	kernel->setArg(2,M); kernel->setArg(3,K);
	kernel->setArg(4,Z); kernel->setArg(5,T);
	kernel->setArg(6,*gam_b); kernel->setArg(7,*alf_b);
	kernel->setArg(8,*bet_b); kernel->setArg(9,*TAU_used_b);
	kernel->setArg(10,*SIG_used_b); kernel->setArg(11,*Otr_b);
	kernel->setArg(12,*MU_used_b); kernel->setArg(13,*gamd_b);
	kernel->setArg(14,*g_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, T, K), cl::NullRange);
	checkErr(err, "k_2_6");

	// кернел 2.7 (set_var)
	kernel = kernels["k_2_7"];
	kernel->setArg(0,N); kernel->setArg(1,K);
	kernel->setArg(2,T); kernel->setArg(3,*ksi_b);
	kernel->setArg(4,*alf_b); kernel->setArg(5,*A_used_b);
	kernel->setArg(6,*B_b); kernel->setArg(7,*bet_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N*N, T1, K), cl::NullRange);
	checkErr(err, "k_2_7");
}

real_t HMM::calcProbability()
{
	// TODO: загрузка массива c from GPU
	cl_int err = queue->enqueueReadBuffer(*c_b, CL_TRUE, 0, T*K*sizeof(real_t), c);	
	checkErr(err, "enqueueReadBuffer() - c_b");
	real_t res=0;
	for(cl_int k=0;k<K;k++)
		for(cl_int t=0;t<T;t++)
			res -= log(c(t,k));
	return res;
}

void HMM::getTestObserv(std::string fname)
{
	std::fstream f;
	f.open(fname,std::fstream::in);
	for(cl_int k=0;k<K;++k)
		for(cl_int t=0;t<T;t++)
			for (cl_int z=0;z<Z;z++)
				f>>Otr(k,t,z);
	f.close();

	// load them to GPU
	cl_int err = queue->enqueueWriteBuffer(*Otr_b, CL_TRUE, 0, K*T*Z*sizeof(real_t), Otr);	
	checkErr(err, "enqueueWritedBuffer() - Otr_b");
}

