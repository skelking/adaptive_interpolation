#include <opencv.hpp>
#include <math.h>

using namespace cv;

//全局变量
int K_value=5;
double T_value=1.15;


int main()
{
	Mat resource=imread("lena.bmp");
	if (!resource.data)
	{
		return 0;
	}

	int W=resource.cols;
	int H=resource.rows;



	Mat dst=Mat(2*H,2*W,CV_8UC3,Scalar(0,0,0));
	
	for (int w=1;w<W;w++)
	{
		for (int h=1;h<H;h++)
		{
			Vec3b color=resource.at<Vec3b>(h,w);
			dst.at<Vec3b>(2*h-1,2*w-1)=color;
		}
	}


		//第一轮插值
	for (int w=6;w<2*W-6;)
	{
		for (int h=6;h<2*H-6;)
		{
			//分别计算45度，135度对角线方向梯度
			//45度

			for (int i=0;i<dst.channels();i++)
			{
				int G1=abs(dst.at<Vec3b>(h+3,w-3)[i]-dst.at<Vec3b>(h+5,w-5)[i])+abs(dst.at<Vec3b>(h+1,w-3)[i]-dst.at<Vec3b>(h+3,w-5)[i])+abs(dst.at<Vec3b>(h-1,w-3)[i]-dst.at<Vec3b>(h+1,w-5)[i])+abs(dst.at<Vec3b>(h-3,w-3)[i]-dst.at<Vec3b>(h-1,w-5)[i])+
					abs(dst.at<Vec3b>(h+3,w-1)[i]-dst.at<Vec3b>(h+5,w-3)[i])+abs(dst.at<Vec3b>(h+1,w-1)[i]-dst.at<Vec3b>(h+3,w-3)[i])+abs(dst.at<Vec3b>(h-1,w-1)[i]-dst.at<Vec3b>(h+1,w-3)[i])+abs(dst.at<Vec3b>(h-3,w-1)[i]-dst.at<Vec3b>(h-1,w-3)[i])+
					abs(dst.at<Vec3b>(h+3,w+1)[i]-dst.at<Vec3b>(h+5,w-1)[i])+abs(dst.at<Vec3b>(h+1,w+1)[i]-dst.at<Vec3b>(h+3,w-1)[i])+abs(dst.at<Vec3b>(h-1,w+1)[i]-dst.at<Vec3b>(h+1,w-1)[i])+abs(dst.at<Vec3b>(h-3,w+1)[i]-dst.at<Vec3b>(h-1,w-1)[i])+
					abs(dst.at<Vec3b>(h+3,w+3)[i]-dst.at<Vec3b>(h+5,w+1)[i])+abs(dst.at<Vec3b>(h+1,w+3)[i]-dst.at<Vec3b>(h+3,w+1)[i])+abs(dst.at<Vec3b>(h-1,w+3)[i]-dst.at<Vec3b>(h+1,w+1)[i])+abs(dst.at<Vec3b>(h-3,w+3)[i]-dst.at<Vec3b>(h-1,w+1)[i]);

				//135度
				int G2=abs(dst.at<Vec3b>(h-3,w-3)[i]-dst.at<Vec3b>(h-5,w-5)[i])+abs(dst.at<Vec3b>(h-1,w-3)[i]-dst.at<Vec3b>(h-3,w-5)[i])+abs(dst.at<Vec3b>(h+1,w-3)[i]-dst.at<Vec3b>(h-1,w-5)[i])+abs(dst.at<Vec3b>(h+3,w-3)[i]-dst.at<Vec3b>(h+1,w-5)[i])+
					abs(dst.at<Vec3b>(h-3,w-1)[i]-dst.at<Vec3b>(h-5,w-3)[i])+abs(dst.at<Vec3b>(h-1,w-1)[i]-dst.at<Vec3b>(h-3,w-3)[i])+abs(dst.at<Vec3b>(h+1,w-1)[i]-dst.at<Vec3b>(h-1,w-3)[i])+abs(dst.at<Vec3b>(h+3,w-1)[i]-dst.at<Vec3b>(h+1,w-3)[i])+
					abs(dst.at<Vec3b>(h-3,w+1)[i]-dst.at<Vec3b>(h-5,w-1)[i])+abs(dst.at<Vec3b>(h-1,w+1)[i]-dst.at<Vec3b>(h-3,w-1)[i])+abs(dst.at<Vec3b>(h+1,w+1)[i]-dst.at<Vec3b>(h-1,w-1)[i])+abs(dst.at<Vec3b>(h+3,w+1)[i]-dst.at<Vec3b>(h+1,w-1)[i])+
					abs(dst.at<Vec3b>(h-3,w+3)[i]-dst.at<Vec3b>(h-5,w+1)[i])+abs(dst.at<Vec3b>(h-1,w+3)[i]-dst.at<Vec3b>(h-3,w+1)[i])+abs(dst.at<Vec3b>(h+1,w+3)[i]-dst.at<Vec3b>(h-1,w+1)[i])+abs(dst.at<Vec3b>(h+3,w+3)[i]-dst.at<Vec3b>(h+1,w+1)[i]);


				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//计算45度方向，135度方向立方卷积插值算出的像素值
				//45度
				int p1=saturate_cast<uchar>(9.0/16.0*(dst.at<Vec3b>(h-1,w+1)[i]+dst.at<Vec3b>(h+1,w-1)[i])-1.0/16.0*(dst.at<Vec3b>(h-3,w+3)[i]+dst.at<Vec3b>(h-3,w-3)[i]));

				//135度
				int p2=saturate_cast<uchar>(9.0/16.0*(dst.at<Vec3b>(h+1,w+1)[i]+dst.at<Vec3b>(h-1,w-1)[i])-1.0/16.0*(dst.at<Vec3b>(h+3,w+3)[i]+dst.at<Vec3b>(h-3,w-3)[i]));


				//自适应插值
				int p_adapt=(w1*p1+w2*p2)/(w1+w2);
				double T1=double(1+G1)/double(1+G2);
				double T2=double(1+G2)/double(1+G1);

				if (G1>G2)
				{
					if (T_value<T1&&T_value>T2)
						dst.at<Vec3b>(h,w)[i]=p2;
					else if(T_value<T2)
						dst.at<Vec3b>(h,w)[i]=p1;
					else
						dst.at<Vec3b>(h,w)[i]=p_adapt;
				}
				else if (G1<G2)
				{
					if (T_value<T2&&T_value>T1)
						dst.at<Vec3b>(h,w)[i]=p1;
					else if (T_value<T1)
						dst.at<Vec3b>(h,w)[i]=p2;
					else
						dst.at<Vec3b>(h,w)[i]=p_adapt;

				}
				else
					dst.at<Vec3b>(h,w)[i]=p_adapt;
			}
				
				
				
			h+=2;
		}
		w+=2;
	}

	
	imwrite("image2.bmp",dst);

	//进行第二次插值

	for (int w=4;w<dst.cols-4;)
	{
		for(int h=3;h<dst.rows-4;)
		{
			for (int i=0;i<dst.channels();i++)
			{
				
				//分别计算水平方向，竖直方向梯度
				//水平方向梯度
				int G1=abs(dst.at<Vec3b>(h-1,w)[i]-dst.at<Vec3b>(h-1,w+2)[i])+abs(dst.at<Vec3b>(h-1,w-2)[i]-dst.at<Vec3b>(h-1,w)[i])+abs(dst.at<Vec3b>(h+1,w)[i]-dst.at<Vec3b>(h+1,w+2)[i])+abs(dst.at<Vec3b>(h+1,w-2)[i]-dst.at<Vec3b>(h+1,w)[i])+
					abs(dst.at<Vec3b>(h,w-1)[i]-dst.at<Vec3b>(h,w+1)[i])+abs(dst.at<Vec3b>(h-2,w-1)[i]-dst.at<Vec3b>(h-2,w+1)[i])+abs(dst.at<Vec3b>(h+2,w-1)[i]-dst.at<Vec3b>(h+2,w+1)[i]);

				//竖直方向梯度
				int G2=abs(dst.at<Vec3b>(h,w-1)[i]-dst.at<Vec3b>(h+2,w-1)[i])+abs(dst.at<Vec3b>(h,w+1)[i]-dst.at<Vec3b>(h+2,w+1)[i])+abs(dst.at<Vec3b>(h-2,w-1)[i]-dst.at<Vec3b>(h,w-1)[i])+abs(dst.at<Vec3b>(h-2,w+1)[i]-dst.at<Vec3b>(h,w+1)[i])+
					abs(dst.at<Vec3b>(h-1,w)[i]-dst.at<Vec3b>(h+1,w)[i])+abs(dst.at<Vec3b>(h-1,w-2)[i]-dst.at<Vec3b>(h+1,w-2)[i])+abs(dst.at<Vec3b>(h-1,w+2)[i]-dst.at<Vec3b>(h+1,w+2)[i]);

				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//计算水平、竖直方向卷积插值算出的像素值
				//水平方向
				int p1=saturate_cast<uchar>(9.0/16.0*(dst.at<Vec3b>(h,w+1)[i]+dst.at<Vec3b>(h,w-1)[i])-1.0/16.0*(dst.at<Vec3b>(h,w+3)[i]+dst.at<Vec3b>(h,w-3)[i]));

				//竖直方向
				int p2=saturate_cast<uchar>(9.0/16.0*(dst.at<Vec3b>(h-1,w)[i]+dst.at<Vec3b>(h+1,w)[i])-1.0/16.0*(dst.at<Vec3b>(h-3,w)[i]+dst.at<Vec3b>(h+3,w)[i]));

				//自适应插值
				int p_adapt1=saturate_cast<uchar>((w1*p1+w2*p2)/(w1+w2));
				double T1=double(1+G1)/double(1+G2);
				double T2=double(1+G2)/double(1+G1);

				if (G1>G2)
				{
					if (T_value<T1&&T_value>T2)
						dst.at<Vec3b>(h,w)[i]=p2;
					else if(T_value<T2)
						dst.at<Vec3b>(h,w)[i]=p1;
					else
						dst.at<Vec3b>(h,w)[i]=p_adapt1;
				}
				else if (G1<G2)
				{
					if (T_value<T2&&T_value>T1)
						dst.at<Vec3b>(h,w)[i]=p1;
					else if (T_value<T1)
						dst.at<Vec3b>(h,w)[i]=p2;
					else
						dst.at<Vec3b>(h,w)[i]=p_adapt1;

				}
				else
					dst.at<Vec3b>(h,w)[i]=p_adapt1;

			}

			h+=2;
		}
		w+=2;
	}

	
	for (int w=3;w<2*W-4;)
	{
		for(int h=4;h<2*H-4;)
		{
			for (int i=0;i<dst.channels();i++)
			{

				//分别计算水平方向，竖直方向梯度
				//水平方向梯度
				int G3=abs(dst.at<Vec3b>(h-1,w)[i]-dst.at<Vec3b>(h-1,w+2)[i])+abs(dst.at<Vec3b>(h-1,w-2)[i]-dst.at<Vec3b>(h-1,w)[i])+abs(dst.at<Vec3b>(h+1,w)[i]-dst.at<Vec3b>(h+1,w+2)[i])+abs(dst.at<Vec3b>(h+1,w-2)[i]-dst.at<Vec3b>(h+1,w)[i])+
					abs(dst.at<Vec3b>(h,w-1)[i]-dst.at<Vec3b>(h,w+1)[i])+abs(dst.at<Vec3b>(h-2,w-1)[i]-dst.at<Vec3b>(h-2,w+1)[i])+abs(dst.at<Vec3b>(h+2,w-1)[i]-dst.at<Vec3b>(h+2,w+1)[i]);

				//竖直方向梯度
				int G4=abs(dst.at<Vec3b>(h,w-1)[i]-dst.at<Vec3b>(h+2,w-1)[i])+abs(dst.at<Vec3b>(h,w+1)[i]-dst.at<Vec3b>(h+2,w+1)[i])+abs(dst.at<Vec3b>(h-2,w-1)[i]-dst.at<Vec3b>(h,w-1)[i])+abs(dst.at<Vec3b>(h-2,w+1)[i]-dst.at<Vec3b>(h,w+1)[i])+
					abs(dst.at<Vec3b>(h-1,w)[i]-dst.at<Vec3b>(h+1,w)[i])+abs(dst.at<Vec3b>(h-1,w-2)[i]-dst.at<Vec3b>(h+1,w-2)[i])+abs(dst.at<Vec3b>(h-1,w+2)[i]-dst.at<Vec3b>(h+1,w+2)[i]);

				double w3=1.0/(1+pow(double(G3),K_value));
				double w4=1.0/(1+pow(double(G4),K_value));

				//计算水平、竖直方向卷积插值算出的像素值
				//水平方向
				int p3=saturate_cast<uchar>(9.0/16.0*(dst.at<Vec3b>(h,w+1)[i]+dst.at<Vec3b>(h,w-1)[i])-1.0/16.0*(dst.at<Vec3b>(h,w+3)[i]+dst.at<Vec3b>(h,w-3)[i]));

				//竖直方向
				int p4=saturate_cast<uchar>(9.0/16.0*(dst.at<Vec3b>(h-1,w)[i]+dst.at<Vec3b>(h+1,w)[i])-1.0/16.0*(dst.at<Vec3b>(h-3,w)[i]+dst.at<Vec3b>(h+3,w)[i]));

				//自适应插值
				int p_adapt2=saturate_cast<uchar>((w3*p3+w4*p4)/(w3+w4));
				double T3=double(1+G3)/double(1+G4);
				double T4=double(1+G4)/double(1+G3);

				if (G3>G4)
				{
					if (T_value<T3&&T_value>T4)
						dst.at<Vec3b>(h,w)[i]=p4;
					else if(T_value<T4)
						dst.at<Vec3b>(h,w)[i]=p3;
					else
						dst.at<Vec3b>(h,w)[i]=p_adapt2;
				}
				else if (G3<G4)
				{
					if (T_value<T4&&T_value>T3)
						dst.at<Vec3b>(h,w)[i]=p3;
					else if (T_value<T3)
						dst.at<Vec3b>(h,w)[i]=p4;
					else
						dst.at<Vec3b>(h,w)[i]=p_adapt2;

				}
				else
					dst.at<Vec3b>(h,w)[i]=p_adapt2;
		
			}
			h+=2;	
		}
		w+=2;
	}

	//锐化
	/*
		0  -1  0
		-1  5  -1
		0  -1  0
	*/
	
	Mat dst_finnal;
	Mat kernal=(Mat_<float>(3,3)<<0,-1,0,
								  -1,5,-1,
								  0,-1,0);
	filter2D(dst,dst_finnal,dst.depth(),kernal,Point(-1,-1));


	imwrite("image_3.bmp",dst_finnal);
	return 1;
}