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

	//ВЫДЕЛЯЕМ ПАМЯТЬ ДЛЯ ПАРАМЕТРОВ СММ
	PI = new real_t[N];					// начальное распределение вероятностей
	A = new real_t[N*N];				// вероятности переходов
	TAU = new real_t[N*M];			
	MU = new real_t[N*M*Z];
	SIG = new real_t[N*M*Z*Z];
	c = new real_t[T*K];				// коэффициенты масштаба
	ksi = new real_t[(T-1)*N*N*K];	
	gam = new real_t[T*N*K];
	gamd = new real_t[T*N*M*K];

	//начальные приближения
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

void HMM::bindOpenCL(cl::Context * context_, std::map<std::string,cl::Kernel*> & kernels_, cl::CommandQueue * queue_) // привязка OpenCL переменных и создание буферов
{
	context = context_; kernels = kernels_; queue = queue_;
	// буферы
	// параметры
	PI_b = new cl::Buffer(*context,NULL,N*sizeof(real_t));
	A_b = new cl::Buffer(*context,NULL,N*N*sizeof(real_t));
	TAU_b = new cl::Buffer(*context,NULL,N*M*sizeof(real_t));
	MU_b = new cl::Buffer(*context,NULL,Z*N*M*sizeof(real_t));
	SIG_b = new cl::Buffer(*context,NULL,Z*Z*N*M*sizeof(real_t));
	// начальные приближения
	PI1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*sizeof(real_t),PI1);
	A1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*N*sizeof(real_t),A1);
	TAU1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*M*sizeof(real_t),TAU1);
	MU1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*M*Z*NumInit*sizeof(real_t),MU1);
	SIG1_b = new cl::Buffer(*context,CL_MEM_COPY_HOST_PTR,N*M*Z*Z*NumInit*sizeof(real_t),SIG1);
	// наблюдения
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

void HMM::allocateDerivatives(int K)
{
	d_PI_b = new cl::Buffer(*context, NULL, K*N*sizeof(real_t));
	d_A_b = new cl::Buffer(*context, NULL, K*N*N*sizeof(real_t));
	d_TAU_b = new cl::Buffer(*context, NULL, K*N*M*sizeof(real_t));
	d_MU_b = new cl::Buffer(*context, NULL, K*Z*N*M*sizeof(real_t));
	d_SIG_b = new cl::Buffer(*context, NULL, K*Z*N*M*sizeof(real_t));

	alf1_zero_b = new cl::Buffer(*context, NULL, N*sizeof(real_t));
	a_zero_b = new cl::Buffer(*context, NULL, N*N*sizeof(real_t));
	b_zero_b = new cl::Buffer(*context, NULL, N*T*sizeof(real_t));
	alf1_PI_b = new cl::Buffer(*context, NULL, K*N*N*sizeof(real_t));
	alf1_MUSIG_b = new cl::Buffer(*context, NULL, K*N*M*Z*N*sizeof(real_t));
	alf1_TAU_b = new cl::Buffer(*context, NULL, K*N*M*N*sizeof(real_t));
	a_A_b = new cl::Buffer(*context, NULL, K*N*N*N*N*sizeof(real_t));
	b_MUSIG_b = new cl::Buffer(*context, NULL, K*N*M*Z*N*T*sizeof(real_t));
	b_TAU_b = new cl::Buffer(*context, NULL, K*N*M*N*T*sizeof(real_t));
	dets_b = new cl::Buffer(*context, NULL, K*N*M*sizeof(real_t));

	cd_b = new cl::Buffer(*context, NULL, K*N*N*M*Z* T*sizeof(real_t));
	alf_t_d_b = new cl::Buffer(*context, NULL, K*N*N*M*Z* T*N*sizeof(real_t));
	alf_s_d_b = new cl::Buffer(*context, NULL, K*N*N*M*Z* T*N*sizeof(real_t));

	// kernel 4.0 - alf1_zero
	cl::Kernel * kernel = kernels["k_4_0"];
	kernel->setArg(0, *alf1_zero_b);
	int err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
	checkErr(err, "k_4_0");
	// kernel 4.0 - a_zero
	kernel = kernels["k_4_0"];
	kernel->setArg(0, *a_zero_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N*N), cl::NullRange);
	checkErr(err, "k_4_0");
	// kernel 4.0 - b_zero
	kernel = kernels["k_4_0"];
	kernel->setArg(0, *b_zero_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N*T), cl::NullRange);
	checkErr(err, "k_4_0");
	// TODO: free derivatives!
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
	// расширения
	//std::string extensionsList;
	//devices[i].getInfo(CL_DEVICE_EXTENSIONS,&extensionsList);		//TODO: включать real_t, если такое расширение доступно
	kernel->getWorkGroupInfo(devices[0],CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,&prefWorkGroupSizeMul);	
}*/

// вспомогательная функция
void copyArray(real_t * dest, real_t * source, cl_int n)
{
	for(cl_int i=0; i<n; i++)
		dest[i]=source[i];
}

void HMM::findModelParameters()
{
	cl_int err;
	// выполним Баума Велша для всех начальных приближений и выбираем лучший набор параметров
	real_t p, p0 = -1000000000000000.;
	// n - номер приближения
	for(cl_int n=0; n<NumInit; n++)
	{
		// оценим по алгоритму Баума-Велша параметры СММ и вернем логарифм вероятности
		p = calcBaumWelсh(n);
		// если данная оценка дала результат лучше - сохраним её
		if(p>p0)
		{
			// load pi,a,tau,mu,sig from gpu and save here
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
	// внутренние вычисления
	internal_calculations(-1);

	// get c from GPU
	cl_int err = queue->enqueueReadBuffer(*c_b, CL_TRUE, 0, T*K*sizeof(real_t), c);	
	checkErr(err, "enqueueReadBuffer() - c_b");

	for(cl_int k = 0; k < K; k++)
		for(cl_int t = 0; t < T; t++)
			p[k] -= log(c(t,k));
}

real_t HMM::calcBaumWelсh(cl_int n)
{
	cl_int err;
	cl_int T1=T-1;
	cl::Event last_event;
	real_t * gam_sum = new real_t[N];
	real_t * gamd_sum = new real_t[N*M];
	// для выхода по вероятности
	double p = 100000000000000000; double p_pred = 1000000000000000000;
	double e = 0.001;
	//std::fstream f; // debug

	// TODO: выход по невязке (по вероятности)
	for(cl_int iter=0; iter<MAX_ITER/*, abs(p-p_pred) > e*/; iter++)
	{
		// большой блок вспомогательных вычислений
		internal_calculations(n);

		cl::Kernel * kernel;

		// кернел 3.1.1
		/*cl::Kernel * kernel = kernels["k_3_1_1"];
		kernel->setArg(0,*gam_sum_b); 
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
		checkErr(err, "k_3_1_1");

		// кернел 3.1.2
		kernel = kernels["k_3_1_2"];
		kernel->setArg(0,M); kernel->setArg(1,*gamd_sum_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, M), cl::NullRange);
		checkErr(err, "k_3_1_2");*/
		
		
		// кернел 3.2 (БОЛЬШОЙ)
		/*cl::Kernel * kernel1 = kernels["k_3_2_1"];
		kernel1->setArg(2,N); kernel1->setArg(3,K);
		kernel1->setArg(4,*gam_sum_b); kernel1->setArg(5,*gam_b);
		kernel1->setArg(6,*flag_b);
		cl::Kernel * kernel2 = kernels["k_3_2_2"];
		kernel2->setArg(2,N); kernel2->setArg(3,M); 
		kernel2->setArg(4,K); kernel2->setArg(5,*gam_sum_b);
		kernel2->setArg(6,*gamd_sum_b); kernel2->setArg(7,*gamd_b);
		kernel2->setArg(8,*flag_b);

		checkErr(err, "enqueueWriteBuffer() - flag_b");*/

		
		// bottleneck
		//#pragma omp parallel for schedule(dynamic, 1)
		for (cl_int i = 0; i < N; i++)
		{
			gam_sum[i] = 0;
			for (cl_int m = 0; m < M; m++)
			{
				gamd_sum[i*M + m] = 0;
			}
		}

		err = queue->enqueueReadBuffer(*gam_b, CL_TRUE, 0, T*N*K*sizeof(real_t), gam);
		err = queue->enqueueReadBuffer(*gamd_b, CL_TRUE, 0, T*N*M*K*sizeof(real_t), gamd);
		//#pragma omp parallel for schedule(dynamic, 1) 
		for (cl_int t = 0; t<T1; t++)
			for (cl_int k = 0; k < K; k++)
				for (cl_int i = 0; i < N; i++)
				{
					gam_sum[i] += gam(t, i, k);
					for (cl_int m = 0; m < M; m++)
					{
						real_t ttt = gamd_sum[i*M + m] + gamd(t, i, m, k);
						if (isfinite(ttt))
							gamd_sum[i*M + m] += gamd(t, i, m, k);
					}
				}
		err = queue->enqueueWriteBuffer(*gam_sum_b, CL_TRUE, 0, N*sizeof(real_t), gam_sum);
		err = queue->enqueueWriteBuffer(*gamd_sum_b, CL_TRUE, 0, N*M*sizeof(real_t), gamd_sum);

		/*for(cl_int t=0; t<T1; t++)
			for(cl_int k=0; k<K; k++)
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
			}
			*/

		// DEBUG - gam_sum, gamd_sum - no error
		/*std::fstream f;
		real_t * gam_sum_dbg = new real_t[N];
		err = queue->enqueueReadBuffer(*gam_sum_b, CL_TRUE, 0, N*sizeof(real_t), gam_sum_dbg);
		checkErr(err, "enqueueReadBuffer() - gam_sum_b");
		f.open("debugging_gam_sum.txt",std::fstream::out);
		for (cl_int i=0; i<N; i++)
			f << gam_sum_dbg[i] << std::endl;
		f.close();
		real_t * gamd_sum_dbg = new real_t[N*M];
		err = queue->enqueueReadBuffer(*gamd_sum_b, CL_TRUE, 0, N*M*sizeof(real_t), gamd_sum_dbg);
		checkErr(err, "enqueueReadBuffer() - gamd_sum_b");
		f.open("debugging_gamd_sum.txt", std::fstream::out);
		for (cl_int i = 0; i<N*M; i++)
			f << gamd_sum_dbg[i] << std::endl;
		f.close();*/
		// /DEBUG
		
		// кернел 3.3
		kernel = kernels["k_3_3"];
		kernel->setArg(0,N); kernel->setArg(1,K); kernel->setArg(2,*PI_b); kernel->setArg(3,*gam_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
		checkErr(err, "k_3_3");

		// DEBUG - PI - good
		/*real_t * PI_dbg = new real_t[N];
		err = queue->enqueueReadBuffer(*PI_b, CL_TRUE, 0, N*sizeof(real_t), PI_dbg);
		checkErr(err, "enqueueReadBuffer() - PI_b");
		f.open("debugging_PI.txt",std::fstream::out);
		for (cl_int i=0; i<N; i++)
		f << PI_dbg[i] << std::endl;
		f.close();*/
		// DEBUG
		
		// кернел 3.4 - bottleneck

		err = queue->enqueueReadBuffer(*ksi_b, CL_TRUE, 0, (T-1)*N*N*K*sizeof(real_t), ksi);
		err = queue->enqueueReadBuffer(*c_b, CL_TRUE, 0, T*K*sizeof(real_t), c);
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				real_t tmp2 = 0.0f;
				for (int k = 0; k < K; k++)							// 2d-редукция
					for (int t = 0; t < T1; t++)
						tmp2 += ksi(t, i, j, k)*c(t + 1, k);
				A(i, j) = tmp2 / gam_sum[i];
			}
		err = queue->enqueueWriteBuffer(*A_b, CL_TRUE, 0, N*N*sizeof(real_t), A);

		/*kernel = kernels["k_3_4"];
		kernel->setArg(0,K); kernel->setArg(1,N); kernel->setArg(2,T1);
		kernel->setArg(3,*A_b); kernel->setArg(4,*ksi_b);
		kernel->setArg(5,*gam_sum_b); kernel->setArg(6,*c_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, N), cl::NullRange);
		checkErr(err, "k_3_4");*/

		// DEBUG - A - good
		/*real_t * A_dbg = new real_t[N*N];
		err = queue->enqueueReadBuffer(*A_b, CL_TRUE, 0, N*N*sizeof(real_t), A_dbg);
		checkErr(err, "enqueueReadBuffer() - A_b");
		f.open("debugging_A.txt",std::fstream::out);
		for (cl_int i=0; i<N*N; i++)
		f << A_dbg[i] << std::endl;
		f.close();*/
		// DEBUG

		// кернел 3.5
		kernel = kernels["k_3_5"];
		kernel->setArg(0,M); kernel->setArg(1,*TAU_b); kernel->setArg(2,*gamd_sum_b);
		kernel->setArg(3,*gam_sum_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, M), cl::NullRange);
		checkErr(err, "k_3_5");

		// DEBUG - TAU - good
		/*real_t * TAU_dbg = new real_t[N*M];
		err = queue->enqueueReadBuffer(*TAU_b, CL_TRUE, 0, N*M*sizeof(real_t), TAU_dbg);
		checkErr(err, "enqueueReadBuffer() - TAU_b");
		f.open("debugging_TAU.txt",std::fstream::out);
		for (cl_int i=0; i<N*M; i++)
			f << TAU_dbg[i] << std::endl;
		f.close();*/
		// DEBUG
		
		// кернел 3.6 - bottleneck
		/*real_t ttt;
		for (int z = 0; z < Z; z++)
		{
			for (int i = 0; i < N; i++)
			{
				for (int m = 0; m < M; m++)
				{
					MU(z, i, m) = 0.0f;
					for (int k = 0; k < K; k++)									// 2d-редукция
						for (int t = 0; t < T; t++)
						{
							ttt = MU(z, i, m) + gamd(t, i, m, k)*Otr(k, t, z);
							if (isfinite(ttt))
								MU(z, i, m) += gamd(t, i, m, k)*Otr(k, t, z);
						}
					MU(z, i, m) /= gamd_sum[i*M + m];
				}
			}
		}
		err = queue->enqueueWriteBuffer(*MU_b, CL_TRUE, 0, Z*N*M*sizeof(real_t), MU);*/

		kernel = kernels["k_3_6"];
		kernel->setArg(0,N); kernel->setArg(1,M); kernel->setArg(2,K);
		kernel->setArg(3,Z); kernel->setArg(4,T); 
		kernel->setArg(5,*MU_b); kernel->setArg(6,*gamd_b); 
		kernel->setArg(7,*Otr_b); kernel->setArg(8,*gamd_sum_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, Z, M), cl::NullRange);
		checkErr(err, "k_3_6");

		// DEBUG - MU - good
		/*real_t * MU_dbg = new real_t[N*M*Z];
		err = queue->enqueueReadBuffer(*MU_b, CL_TRUE, 0, N*M*Z*sizeof(real_t), MU_dbg);
		checkErr(err, "enqueueReadBuffer() - MU_b");
		f.open("debugging_MU.txt",std::fstream::out);
		for (cl_int i=0; i<N*M*Z; i++)
		f << MU_dbg[i] << std::endl;
		f.close();*/
		// DEBUG

		// кернел 3.7 - bottleneck
		/*real_t tmp3;
		for (cl_int z1 = 0; z1<Z; z1++)
			for (cl_int z2 = 0; z2 < Z; z2++)
			{
				for (cl_int i = 0; i < N; i++)
				{
					for (cl_int m = 0; m < M; m++)
					{
						SIG(z1, z2, i, m) = 0.0f;
						for (cl_int k = 0; k < K; k++)										// 2d-редукция
							for (cl_int t = 0; t < T; t++)
							{
								tmp3 = (Otr(k, t, z1) - MU(z1, i, m)) * (Otr(k, t, z2) - MU(z2, i, m));
								ttt = SIG(z1, z2, i, m) + gamd(t, i, m, k)*tmp3;
								if (isfinite(ttt))
									SIG(z1, z2, i, m) += gamd(t, i, m, k)*tmp3;
							}
						SIG(z1, z2, i, m) /= gamd_sum[i*M + m];
					}
				}
			}
		err = queue->enqueueWriteBuffer(*SIG_b, CL_TRUE, 0, Z*Z*N*M*sizeof(real_t), SIG);*/

		kernel = kernels["k_3_7"];
		kernel->setArg(0,N); kernel->setArg(1,M); kernel->setArg(2,Z);
		kernel->setArg(3,K); kernel->setArg(4,T);
		kernel->setArg(5,*SIG_b); kernel->setArg(6,*gamd_b); 
		kernel->setArg(7,*gamd_sum_b); 
		kernel->setArg(8,*MU_b); kernel->setArg(9,*Otr_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, Z*Z, M), cl::NullRange);
		checkErr(err, "k_3_7");

		// DEBUG - SIG - good
		/*real_t * SIG_dbg = new real_t[N*M*Z*Z];
		err = queue->enqueueReadBuffer(*SIG_b, CL_TRUE, 0, N*M*Z*Z*sizeof(real_t), SIG_dbg);
		checkErr(err, "enqueueReadBuffer() - SIG_b");
		f.open("debugging_SIG.txt",std::fstream::out);
		for (cl_int i=0; i<N*M*Z*Z; i++)
			f << SIG_dbg[i] << std::endl;
		f.close();*/
		// DEBUG
		
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

		// DEBUG - SIG1 - good
		/*std::fstream f;
		real_t * SIG1_dbg = new real_t[N*M*Z*Z];
		err = queue->enqueueReadBuffer(*SIG1_b, CL_TRUE, 0, N*M*Z*Z*sizeof(real_t), SIG1_dbg);
		checkErr(err, "enqueueReadBuffer() - SIG_b");
		f.open("debugging_SIG1.txt", std::fstream::out);
		for (cl_int i = 0; i<N*M*Z*Z; i++)
			f << SIG1_dbg[i] << std::endl;
		f.close();*/
		// DEBUG
		// посчитаем новый логарифм вероятности
		p = calcProbability();
		std::swap(p, p_pred);
	}

	delete gam_sum;
	delete gamd_sum;
	
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

	// кернел calc_G (заранее считаем все g)
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

	// DEBUG - satisfying
	/*real_t * B_dbg = new real_t[N*T*K];
	err = queue->enqueueReadBuffer(*B_b, CL_TRUE, 0, N*T*K*sizeof(real_t), B_dbg);	
	checkErr(err, "enqueueReadBuffer() - B_b");
	std::fstream f;
	f.open("debugging_B.txt",std::fstream::out);
	for (cl_int i=0; i<N*T*K; i++)
		f << B_dbg[i] << std::endl;
	f.close();*/
	// /DEBUG

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

	// DEBUG alf, alf_t - satisfying
	/*real_t * alf_dbg = new real_t[N*T*K];
	err = queue->enqueueReadBuffer(*alf_b, CL_TRUE, 0, N*T*K*sizeof(real_t), alf_dbg);	
	checkErr(err, "enqueueReadBuffer() - alf_b");
	real_t * alf_t_dbg = new real_t[N*T*K];
	err = queue->enqueueReadBuffer(*alf_t_b, CL_TRUE, 0, N*T*K*sizeof(real_t), alf_t_dbg);
	checkErr(err, "enqueueReadBuffer() - alf_t_b");
	std::fstream f;
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
	
	// DEBUG c - satisfying
	/*real_t * c_dbg = new real_t[T*K];
	err = queue->enqueueReadBuffer(*c_b, CL_TRUE, 0, T*K*sizeof(real_t), c_dbg);
	checkErr(err, "enqueueReadBuffer() - c_b");
	std::fstream f;
	f.open("debugging_c.txt",std::fstream::out);
	for (cl_int i=0; i<T*K; i++)
	f << c_dbg[i] << std::endl;
	f.close();*/
	// /DEBUG

	///
	/// далее вычисления только для этапа обучения
	///
	if (n != -1) 
	{
		// кернел 2.5 (большой)
		kernel1 = kernels["k_2_5_1"];
		kernel1->setArg(1, N);
		kernel1->setArg(2, K); kernel1->setArg(3, *bet_t_b);
		kernel1->setArg(4, *c_b); kernel1->setArg(5, *bet_b);
		kernel2 = kernels["k_2_5_2"];
		kernel2->setArg(1, N); kernel2->setArg(2, K);
		kernel2->setArg(3, T); kernel2->setArg(4, *bet_b);
		kernel2->setArg(5, *A_used_b); kernel2->setArg(6, *B_b);
		kernel2->setArg(7, *bet_t_b); kernel2->setArg(8, *alf_b);
		for (cl_int t = T1 - 1; t >= 0; t--)
		{
			// кернел 2.5.1
			kernel1->setArg(0, t);
			err = queue->enqueueNDRangeKernel(*kernel1, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
			checkErr(err, "k_2_5_1");
			// кернел 2.5.2
			kernel2->setArg(0, t);
			err = queue->enqueueNDRangeKernel(*kernel2, cl::NullRange, cl::NDRange(N, K), cl::NullRange);
			checkErr(err, "k_2_5_2");
		}

		// DEBUG bet, bet_t - satisfying
		/*std::fstream f;
		real_t * bet_dbg = new real_t[N*T*K];
		err = queue->enqueueReadBuffer(*bet_b, CL_TRUE, 0, N*T*K*sizeof(real_t), bet_dbg);
		checkErr(err, "enqueueReadBuffer() - bet_b");
		f.open("debugging_bet.txt",std::fstream::out);
		for (cl_int i=0; i<N*T*K; i++)
		f << bet_dbg[i] << std::endl;
		f.close();
		real_t * bet_t_dbg = new real_t[N*T*K];
		err = queue->enqueueReadBuffer(*bet_t_b, CL_TRUE, 0, N*T*K*sizeof(real_t), bet_t_dbg);
		checkErr(err, "enqueueReadBuffer() - bet_t_b");
		f.open("debugging_bet_t.txt", std::fstream::out);
		for (cl_int i = 0; i<N*T*K; i++)
		f << bet_t_dbg[i] << std::endl;
		f.close();*/
		// /DEBUG

		// кернел 2.6 
		kernel = kernels["k_2_6"];
		kernel->setArg(0, n); kernel->setArg(1, N);
		kernel->setArg(2, M); kernel->setArg(3, K);
		kernel->setArg(4, Z); kernel->setArg(5, T);
		kernel->setArg(6, *gam_b); kernel->setArg(7, *alf_b);
		kernel->setArg(8, *bet_b); kernel->setArg(9, *TAU_used_b);
		kernel->setArg(10, *SIG_used_b); kernel->setArg(11, *Otr_b);
		kernel->setArg(12, *MU_used_b); kernel->setArg(13, *gamd_b);
		kernel->setArg(14, *g_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N, T, K), cl::NullRange);
		checkErr(err, "k_2_6");

		// DEBUG gam, gamd - satisfying
		/*std::fstream f;
		real_t * gam_dbg = new real_t[N*T*K];
		err = queue->enqueueReadBuffer(*gam_b, CL_TRUE, 0, N*T*K*sizeof(real_t), gam_dbg);
		checkErr(err, "enqueueReadBuffer() - gam_b");
		f.open("debugging_gam.txt",std::fstream::out);
		for (cl_int i=0; i<N*T*K; i++)
		f << gam_dbg[i] << std::endl;
		f.close();
		real_t * gamd_dbg = new real_t[N*M*T*K];
		err = queue->enqueueReadBuffer(*gamd_b, CL_TRUE, 0, N*M*T*K*sizeof(real_t), gamd_dbg);
		checkErr(err, "enqueueReadBuffer() - gamd_b");
		f.open("debugging_gamd.txt", std::fstream::out);
		for (cl_int i = 0; i<N*M*T*K; i++)
		f << gamd_dbg[i] << std::endl;
		f.close();*/
		// /DEBUG

		// кернел 2.7 (set_var)
		kernel = kernels["k_2_7"];
		kernel->setArg(0, N); kernel->setArg(1, K);
		kernel->setArg(2, T); kernel->setArg(3, *ksi_b);
		kernel->setArg(4, *alf_b); kernel->setArg(5, *A_used_b);
		kernel->setArg(6, *B_b); kernel->setArg(7, *bet_b);
		err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N*N, T1, K), cl::NullRange);
		checkErr(err, "k_2_7");

		// DEBUG ksi - satisfying
		/*std::fstream f;
		real_t * ksi_dbg = new real_t[N*N*T1*K];
		err = queue->enqueueReadBuffer(*ksi_b, CL_TRUE, 0, N*N*T1*K*sizeof(real_t), ksi_dbg);
		checkErr(err, "enqueueReadBuffer() - ksi_b");
		f.open("debugging_ksi.txt",std::fstream::out);
		for (cl_int i=0; i<N*N*T1*K; i++)
		f << ksi_dbg[i] << std::endl;
		f.close();*/
		// /DEBUG
	}
}

real_t HMM::calcProbability()
{
	//printf("point-1\n");
	// TODO: загрузка массива c from GPU
	//queue->flush();
	//queue->finish();
	//cl_int err = queue->enqueueReadBuffer(*c_b, CL_TRUE, 0, T*K*sizeof(real_t), c);	
	//checkErr(err, "enqueueReadBuffer() - c_b");
	real_t res=0;
	for(cl_int k=0;k<K;k++)
		for(cl_int t=0;t<T;t++)
			res -= log(c(t,k));
	return res;
}

void HMM::getObservations(std::string fname)
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

void HMM::getObservations(std::string fname, real_t * Otr)
{
	std::fstream f;
	f.open(fname, std::fstream::in);
	for (int k = 0; k<K; ++k)
		for (int t = 0; t<T; t++)
			for (int z = 0; z<Z; z++)
				f >> Otr(k, t, z);
	f.close();
}

// for debug means
void saveDerivativesToFile(std::string fileName, int N, int M, int Z, int K, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG)
{
	std::ofstream f;
	f.open(fileName, std::fstream::out);
	f << "Model" << std::endl;
	for (int k = 0; k < K; k++)
	{
		f << "k = " << k << std::endl;
		f << "PI " << std::endl;
		for (int i = 0, size = 2 * N; i < size; i++)
			f << d_PI[k*size + i] << " ";
		f << std::endl;
		f << "A " << std::endl;
		for (int i = 0, size = 2 * N*N; i < size; i++)
			f << d_A[k*size + i] << " ";
		f << std::endl;
		f << "TAU " << std::endl;
		for (int i = 0, size = 2 * N*M; i < size; i++)
			f << d_TAU[k*size + i] << " ";
		f << std::endl;
		f << "MU " << std::endl;
		for (int i = 0, size = 2 * Z*N*M; i < size; i++)
			f << d_MU[k*size + i] << " ";
		f << std::endl;
		f << "SIG " << std::endl;
		for (int i = 0, size = 2 * Z*N*M; i < size; i++){
			f << d_SIG[k*size + i] << " ";
		}
		f << std::endl;
	}
	f.close();
}

svm_model * HMM::trainWithDerivatives(real_t ** observations, int K, HMM ** models, int numModels, svm_scaling_parameters & scalingParams)
{
	int N = models[0]->N, M = models[0]->M, Z = models[0]->Z, T = models[0]->T;	// all the same
	// allocate memory for training derivatives for model 1 and 2
	real_t ** d_PI = new real_t *[numModels];
	real_t ** d_A = new real_t *[numModels];
	real_t ** d_TAU = new real_t *[numModels];
	real_t ** d_MU = new real_t *[numModels];
	real_t ** d_SIG = new real_t *[numModels];
	d_PI[0] = new real_t[2*K*N];
	d_A[0] = new real_t[2*K*N*N];
	d_TAU[0] = new real_t[2*K*N*M];
	d_MU[0] = new real_t[2*K*Z*N*M];
	d_SIG[0] = new real_t[2*K*Z*N*M];
	d_PI[1] = new real_t[2*K*N];
	d_A[1] = new real_t[2*K*N*N];
	d_TAU[1] = new real_t[2*K*N*M];
	d_MU[1] = new real_t[2*K*Z*N*M];
	d_SIG[1] = new real_t[2*K*Z*N*M];

	// TODO: allocate memory on GPU for derivatives and misc
	models[0]->allocateDerivatives(K);
	models[1]->allocateDerivatives(K);

	/// таймер
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	double elapsedTime;
	QueryPerformanceFrequency(&frequency);  // get ticks per second
	/// таймер
	QueryPerformanceCounter(&t1);				// start timer

	// calculate derivatives for 1st model and 1st training observations
	models[0]->calcDerivatives(observations[0], K, &d_PI[0][0], &d_A[0][0], &d_TAU[0][0], &d_MU[0][0], &d_SIG[0][0]);
	// calculate derivatives for 1st model and 2nd training observations
	models[0]->calcDerivatives(observations[1], K, &d_PI[0][K*N], &d_A[0][K*N*N], &d_TAU[0][K*N*M], &d_MU[0][K*Z*N*M], &d_SIG[0][K*Z*N*M]);
	// calculate derivatives for 2nd model and 1st training observations
	models[1]->calcDerivatives(observations[0], K, &d_PI[1][0], &d_A[1][0], &d_TAU[1][0], &d_MU[1][0], &d_SIG[1][0]);
	// calculate derivatives for 2st model and 2nd training observations
	models[1]->calcDerivatives(observations[1], K, &d_PI[1][K*N], &d_A[1][K*N*N], &d_TAU[1][K*N*M], &d_MU[1][K*Z*N*M], &d_SIG[1][K*Z*N*M]);
	
	QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Derivatives learning complete\nElapsed time = %f s.\n", elapsedTime);

	// debug
	saveDerivativesToFile("M1_train_derivs.txt", N, M, Z, K, d_PI[0], d_A[0], d_TAU[0], d_MU[0], d_SIG[0]);
	saveDerivativesToFile("M2_train_derivs.txt", N, M, Z, K, d_PI[1], d_A[1], d_TAU[1], d_MU[1], d_SIG[1]);
	// debug

	// SVM training
	svm_model model;

	// TODO: free memory
	return &model;
}

void HMM::calcDerivatives(real_t * observations, int nOfSequences, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG)
{
	real_t * old_Otr = Otr;
	int old_K = K;
	Otr = observations;
	K = nOfSequences;
	cl::Buffer * old_Otr_b = Otr_b;

	// TODO: resize B, g, etc. accordingly to nOfSequences (K)

	// load observations to GPU
	Otr_b = new cl::Buffer(*context, CL_MEM_COPY_HOST_PTR, K*T*Z*sizeof(real_t), Otr);

	// carry out some internal calculations 
	internal_calculations(-1);

	// calc derivative for each parameter and each sequence
	calc_derivatives_for_all_sequences();

	// load calculated derivatives to cpu stored derivatives
	cl_int err = queue->enqueueReadBuffer(*d_PI_b, CL_TRUE, 0, K*N*sizeof(real_t), d_PI);
	checkErr(err, "enqueueReadBuffer() - d_PI_b");
	err = queue->enqueueReadBuffer(*d_A_b, CL_TRUE, 0, K*N*N*sizeof(real_t), d_A);
	checkErr(err, "enqueueReadBuffer() - d_A_b");
	err = queue->enqueueReadBuffer(*d_TAU_b, CL_TRUE, 0, K*N*M*sizeof(real_t), d_TAU);
	checkErr(err, "enqueueReadBuffer() - d_TAU_b");
	err = queue->enqueueReadBuffer(*d_MU_b, CL_TRUE, 0, K*Z*N*M*sizeof(real_t), d_MU);
	checkErr(err, "enqueueReadBuffer() - d_MU_b");
	err = queue->enqueueReadBuffer(*d_SIG_b, CL_TRUE, 0, K*Z*N*M*sizeof(real_t), d_SIG);
	checkErr(err, "enqueueReadBuffer() - d_SIG_b");

	delete Otr_b;	// delete processed observations
	Otr = old_Otr;
	K = old_K;
	Otr_b = old_Otr_b;
}

void HMM::calc_derivatives_for_all_sequences()
{
	int err;
//#define d_A(k,i,j) d_A[((k)*N+i)*N+j]
//#define d_TAU(k,i,m) d_TAU[((k)*N+i)*M+m]
//#define d_MU(k,z,i,m) d_MU[(((k)*N+i)*M+m)*Z+z]
//#define d_SIG(k,z,i,m) d_SIG[(((k)*N+i)*M+m)*Z+z]
//#define dets(k,i,m) dets[((k)*N+i)*M+m]
//#define a_A(k,i,j,i1,j1) a_A[((((k)*N+i)*N+j)*N+i1)*N+j1]
//#define b_MUSIG(k,i,m,z,i1,t) b_MUSIG[(((((k)*N+i)*M+m)*Z+z)*N+i1)*T+t]
//#define b_TAU(k, i, m, i1, t) b_TAU[((((k)*N+i)*M+m)*N+i1)*T+t]
//#define alf1_PI(k,i,j) alf1_PI[((k)*N+i)*N+j]
//#define alf1_MUSIG(k, i, m, z, i1) alf1_MUSIG[((((k)*N+i)*M+m)*Z+z)*N+i1]
//#define alf1_TAU(k, i, m, i1) alf1_TAU[(((k)*N+i)*M+m)*N+i1]

	// derivatives with respect to PI 
	cl::Kernel * kernel;
	// kernel 4.1.1 KxNxN
	kernel = kernels["k_4_1_1"];
	kernel->setArg(0, N); kernel->setArg(1, T); kernel->setArg(2, K);
	kernel->setArg(3, *alf1_PI_b); kernel->setArg(4, *B_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, N), cl::NullRange);
	checkErr(err, "k_4_1_1");

	// kernel 4.1.2 KxN
	kernel = kernels["k_4_1_2"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *alf1_PI_b); kernel->setArg(6, *a_zero_b); kernel->setArg(7, *b_zero_b);
	kernel->setArg(8, *cd_b); kernel->setArg(9, *alf_t_d_b); kernel->setArg(10, *alf_s_d_b);
	kernel->setArg(11, *c_b); kernel->setArg(12, *alf_b); kernel->setArg(13, *alf_t_b);
	kernel->setArg(14, *A_b); kernel->setArg(15, *B_b); kernel->setArg(16, *d_PI_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N), cl::NullRange);
	checkErr(err, "k_4_1_2");

	// derivatives with respect to A 
	// kernel 4.2.1 (KxNxNxNxN)
	kernel = kernels["k_4_2_1"];
	kernel->setArg(0, N); kernel->setArg(1, *a_A_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, N), cl::NullRange);
	checkErr(err, "k_4_2_1");

	// kernel 4.2.2 (KxNxN)
	kernel = kernels["k_4_2_2"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *alf1_zero_b); kernel->setArg(6, *a_A_b); kernel->setArg(7, *b_zero_b);
	kernel->setArg(8, *cd_b); kernel->setArg(9, *alf_t_d_b); kernel->setArg(10, *alf_s_d_b);
	kernel->setArg(11, *c_b); kernel->setArg(12, *alf_b); kernel->setArg(13, *alf_t_b);
	kernel->setArg(14, *A_b); kernel->setArg(15, *B_b); kernel->setArg(16, *d_A_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, N), cl::NullRange);
	checkErr(err, "k_4_2_2");


	// derivatives with respect to MU
	// kernel 4.3.1 (KxNxMxZxNxT)
	kernel = kernels["k_4_3_1"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K); 
	kernel->setArg(5, *b_MUSIG_b); kernel->setArg(6, *TAU_b); kernel->setArg(7, *g_b); kernel->setArg(8, *Otr_b);
	kernel->setArg(9, *MU_b); kernel->setArg(10, *SIG_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, T, N), cl::NullRange);
	checkErr(err, "k_4_3_1");

	// kernel 4.3.2 (KxNxMxZxN)
	kernel = kernels["k_4_3_2"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T);
	kernel->setArg(4, *alf1_MUSIG_b); kernel->setArg(5, *PI_b); kernel->setArg(6, *b_MUSIG_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, M), cl::NullRange);
	checkErr(err, "k_4_3_2");

	// kernel 4.3.3 (KxZxNxM)
	kernel = kernels["k_4_3_3"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *alf1_MUSIG_b); kernel->setArg(6, *a_zero_b); kernel->setArg(7, *b_MUSIG_b);
	kernel->setArg(8, *cd_b); kernel->setArg(9, *alf_t_d_b); kernel->setArg(10, *alf_s_d_b);
	kernel->setArg(11, *c_b); kernel->setArg(12, *alf_b); kernel->setArg(13, *alf_t_b);
	kernel->setArg(14, *A_b); kernel->setArg(15, *B_b); kernel->setArg(16, *d_MU_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, M*Z), cl::NullRange);
	checkErr(err, "k_4_3_3");

	// derivatives with respect to SIG
	// kernel 4.4.1 (KxNxM)
	kernel = kernels["k_4_4_1"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z);
	kernel->setArg(3, *dets_b); kernel->setArg(4, *SIG_b); 
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, M), cl::NullRange);
	checkErr(err, "k_4_4_1");

	// kernel 4.4.2 (KxNxMxZxN)
	kernel = kernels["k_4_4_2"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *b_MUSIG_b); kernel->setArg(6, *TAU_b); kernel->setArg(7, *g_b); kernel->setArg(8, *Otr_b);
	kernel->setArg(9, *MU_b); kernel->setArg(10, *SIG_b); kernel->setArg(11, *dets_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, T, N), cl::NullRange);
	checkErr(err, "k_4_4_2");

	// kernel 4.4.3 (KxNxMxZxN)
	kernel = kernels["k_4_4_3"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T);
	kernel->setArg(4, *alf1_MUSIG_b); kernel->setArg(5, *PI_b); kernel->setArg(6, *b_MUSIG_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, M), cl::NullRange);
	checkErr(err, "k_4_4_3");

	// kernel 4.4.4 (KxZxNxM)
	kernel = kernels["k_4_4_4"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *alf1_MUSIG_b); kernel->setArg(6, *a_zero_b); kernel->setArg(7, *b_MUSIG_b);
	kernel->setArg(8, *cd_b); kernel->setArg(9, *alf_t_d_b); kernel->setArg(10, *alf_s_d_b);
	kernel->setArg(11, *c_b); kernel->setArg(12, *alf_b); kernel->setArg(13, *alf_t_b);
	kernel->setArg(14, *A_b); kernel->setArg(15, *B_b); kernel->setArg(16, *d_SIG_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, M*Z), cl::NullRange);
	checkErr(err, "k_4_4_4");

	// derivatives with respect to TAU
	// kernel 4.5.1 (KxNxMxNxT)
	kernel = kernels["k_4_5_1"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *b_TAU_b); kernel->setArg(6, *g_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, T, N), cl::NullRange);
	checkErr(err, "k_4_5_1");

	// kernel 4.5.2 (KxNxMxN)
	kernel = kernels["k_4_5_2"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *alf1_TAU_b); kernel->setArg(6, *PI_b); kernel->setArg(7, *b_TAU_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, M), cl::NullRange);
	checkErr(err, "k_4_5_2");

	// kernel 4.5.3 (KxNxM)
	kernel = kernels["k_4_5_3"];
	kernel->setArg(0, N); kernel->setArg(1, M); kernel->setArg(2, Z); kernel->setArg(3, T); kernel->setArg(4, K);
	kernel->setArg(5, *alf1_TAU_b); kernel->setArg(6, *a_zero_b); kernel->setArg(7, *b_TAU_b);
	kernel->setArg(8, *cd_b); kernel->setArg(9, *alf_t_d_b); kernel->setArg(10, *alf_s_d_b);
	kernel->setArg(11, *c_b); kernel->setArg(12, *alf_b); kernel->setArg(13, *alf_t_b);
	kernel->setArg(14, *A_b); kernel->setArg(15, *B_b); kernel->setArg(16, *d_TAU_b);
	err = queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(K, N, M), cl::NullRange);
	checkErr(err, "k_4_5_3");
}
