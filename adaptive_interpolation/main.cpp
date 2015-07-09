#include <opencv.hpp>
#include <math.h>

using namespace cv;

//全局变量
int K_value=5;
double T_value=1.15;


int main()
{
	Mat resource=imread("lena.jpg");
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

	//将彩色图像分解为3个通道
	Mat* BGR=new Mat[3];
	split(dst,BGR);



	for(int i=0;i<3;i++)
	{
		Mat tempMat=BGR[i];
		//第一轮插值
		for (int w=6;w<2*W-6;)
		{
			for (int h=6;h<2*H-6;)
			{
				//分别计算45度，135度对角线方向梯度
				//45度
				
				int G1=abs(tempMat.at<uchar>(h+3,w-3)-tempMat.at<uchar>(h+5,w-5))+abs(tempMat.at<uchar>(h+1,w-3)-tempMat.at<uchar>(h+3,w-5))+abs(tempMat.at<uchar>(h-1,w-3)-tempMat.at<uchar>(h+1,w-5))+abs(tempMat.at<uchar>(h-3,w-3)-tempMat.at<uchar>(h-1,w-5))+
					   abs(tempMat.at<uchar>(h+3,w-1)-tempMat.at<uchar>(h+5,w-3))+abs(tempMat.at<uchar>(h+1,w-1)-tempMat.at<uchar>(h+3,w-3))+abs(tempMat.at<uchar>(h-1,w-1)-tempMat.at<uchar>(h+1,w-3))+abs(tempMat.at<uchar>(h-3,w-1)-tempMat.at<uchar>(h-1,w-3))+
					   abs(tempMat.at<uchar>(h+3,w+1)-tempMat.at<uchar>(h+5,w-1))+abs(tempMat.at<uchar>(h+1,w+1)-tempMat.at<uchar>(h+3,w-1))+abs(tempMat.at<uchar>(h-1,w+1)-tempMat.at<uchar>(h+1,w-1))+abs(tempMat.at<uchar>(h-3,w+1)-tempMat.at<uchar>(h-1,w-1))+
					   abs(tempMat.at<uchar>(h+3,w+3)-tempMat.at<uchar>(h+5,w+1))+abs(tempMat.at<uchar>(h+1,w+3)-tempMat.at<uchar>(h+3,w+1))+abs(tempMat.at<uchar>(h-1,w+3)-tempMat.at<uchar>(h+1,w+1))+abs(tempMat.at<uchar>(h-3,w+3)-tempMat.at<uchar>(h-1,w+1));
				
				//135度
				int G2=abs(tempMat.at<uchar>(h-3,w-3)-tempMat.at<uchar>(h-5,w-5))+abs(tempMat.at<uchar>(h-1,w-3)-tempMat.at<uchar>(h-3,w-5))+abs(tempMat.at<uchar>(h+1,w-3)-tempMat.at<uchar>(h-1,w-5))+abs(tempMat.at<uchar>(h+3,w-3)-tempMat.at<uchar>(h+1,w-5))+
					   abs(tempMat.at<uchar>(h-3,w-1)-tempMat.at<uchar>(h-5,w-3))+abs(tempMat.at<uchar>(h-1,w-1)-tempMat.at<uchar>(h-3,w-3))+abs(tempMat.at<uchar>(h+1,w-1)-tempMat.at<uchar>(h-1,w-3))+abs(tempMat.at<uchar>(h+3,w-1)-tempMat.at<uchar>(h+1,w-3))+
					   abs(tempMat.at<uchar>(h-3,w+1)-tempMat.at<uchar>(h-5,w-1))+abs(tempMat.at<uchar>(h-1,w+1)-tempMat.at<uchar>(h-3,w-1))+abs(tempMat.at<uchar>(h+1,w+1)-tempMat.at<uchar>(h-1,w-1))+abs(tempMat.at<uchar>(h+3,w+1)-tempMat.at<uchar>(h+1,w-1))+
					   abs(tempMat.at<uchar>(h-3,w+3)-tempMat.at<uchar>(h-5,w+1))+abs(tempMat.at<uchar>(h-1,w+3)-tempMat.at<uchar>(h-3,w+1))+abs(tempMat.at<uchar>(h+1,w+3)-tempMat.at<uchar>(h-1,w+1))+abs(tempMat.at<uchar>(h+3,w+3)-tempMat.at<uchar>(h+1,w+1));

				
				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//计算45度方向，135度方向立方卷积插值算出的像素值
				//45度
				int p1=9.0/16.0*(tempMat.at<uchar>(h-1,w+1)+tempMat.at<uchar>(h+1,w-1))-1.0/16.0*(tempMat.at<uchar>(h-3,w+3)+tempMat.at<uchar>(h-3,w-3));
				
				//135度
				int p2=9.0/16.0*(tempMat.at<uchar>(h+1,w+1)+tempMat.at<uchar>(h-1,w-1))-1.0/16.0*(tempMat.at<uchar>(h+3,w+3)+tempMat.at<uchar>(h-3,w-3));

				p1=p1>255?255:p1;
				p2=p2>255?255:p2;

				//自适应插值
				int p_adapt=(w1*p1+w2*p2)/(w1+w2);
				double T1=double(1+G1)/double(1+G2);
				double T2=double(1+G2)/double(1+G1);

				if (G1>G2)
				{
					if (T_value<T1&&T_value>T2)
						BGR[i].at<uchar>(h,w)=p2;
					else if(T_value<T2)
						BGR[i].at<uchar>(h,w)=p1;
					else
						BGR[i].at<uchar>(h,w)=p_adapt;
				}
				else if (G1<G2)
				{
					if (T_value<T2&&T_value>T1)
						BGR[i].at<uchar>(h,w)=p1;
					else if (T_value<T1)
						BGR[i].at<uchar>(h,w)=p2;
					else
						BGR[i].at<uchar>(h,w)=p_adapt;
					
				}
				else
					BGR[i].at<uchar>(h,w)=p_adapt;
				
				h+=2;
			}
			w+=2;
		}

	}

	/*Mat image_2;
	merge(BGR,3,image_2);
	imwrite("image2.bmp",image_2);*/

	//进行第二次插值
	for (int i=0;i<3;i++)
	{
		Mat tempMat=BGR[i];
		for (int w=4;w<2*W-4;)
		{
			for(int h=3;h<2*H-4;)
			{
				//分别计算水平方向，竖直方向梯度
				//水平方向梯度
				int G1=abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h-1,w+2))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h-1,w))+abs(tempMat.at<uchar>(h+1,w)-tempMat.at<uchar>(h+1,w+2))+abs(tempMat.at<uchar>(h+1,w-2)-tempMat.at<uchar>(h+1,w))+
				       abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h-2,w+1))+abs(tempMat.at<uchar>(h+2,w-1)-tempMat.at<uchar>(h+2,w+1));
				
				//竖直方向梯度
				int G2=abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h+2,w-1))+abs(tempMat.at<uchar>(h,w+1)-tempMat.at<uchar>(h+2,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h,w-1))+abs(tempMat.at<uchar>(h-2,w+1)-tempMat.at<uchar>(h,w+1))+
					   abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h+1,w))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h+1,w-2))+abs(tempMat.at<uchar>(h-1,w+2)-tempMat.at<uchar>(h+1,w+2));

				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//计算水平、竖直方向卷积插值算出的像素值
				//水平方向
				int p1=9.0/16.0*(tempMat.at<uchar>(h,w+1)+tempMat.at<uchar>(h,w-1))-1.0/16.0*(tempMat.at<uchar>(h,w+3)+tempMat.at<uchar>(h,w-3));

				//竖直方向
				int p2=9.0/16.0*(tempMat.at<uchar>(h-1,w)+tempMat.at<uchar>(h+1,w))-1.0/16.0*(tempMat.at<uchar>(h-3,w)+tempMat.at<uchar>(h+3,w));

				//自适应插值
				int p_adapt=(w1*p1+w2*p2)/(w1+w2);
				double T1=double(1+G1)/double(1+G2);
				double T2=double(1+G2)/double(1+G1);

				if (G1>G2)
				{
					if (T_value<T1&&T_value>T2)
						BGR[i].at<uchar>(h,w)=p2;
					else if(T_value<T2)
						BGR[i].at<uchar>(h,w)=p1;
					else
						BGR[i].at<uchar>(h,w)=p_adapt;
				}
				else if (G1<G2)
				{
					if (T_value<T2&&T_value>T1)
						BGR[i].at<uchar>(h,w)=p1;
					else if (T_value<T1)
						BGR[i].at<uchar>(h,w)=p2;
					else
						BGR[i].at<uchar>(h,w)=p_adapt;

				}
				else
					BGR[i].at<uchar>(h,w)=p_adapt;


				h+=2;
			}
			w+=2;
		}


		for (int w=3;w<2*W-4;)
		{
			for(int h=4;h<2*H-4;)
			{
				//分别计算水平方向，竖直方向梯度
				//水平方向梯度
				int G1=abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h-1,w+2))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h-1,w))+abs(tempMat.at<uchar>(h+1,w)-tempMat.at<uchar>(h+1,w+2))+abs(tempMat.at<uchar>(h+1,w-2)-tempMat.at<uchar>(h+1,w))+
					abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h-2,w+1))+abs(tempMat.at<uchar>(h+2,w-1)-tempMat.at<uchar>(h+2,w+1));

				//竖直方向梯度
				int G2=abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h+2,w-1))+abs(tempMat.at<uchar>(h,w+1)-tempMat.at<uchar>(h+2,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h,w-1))+abs(tempMat.at<uchar>(h-2,w+1)-tempMat.at<uchar>(h,w+1))+
					abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h+1,w))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h+1,w-2))+abs(tempMat.at<uchar>(h-1,w+2)-tempMat.at<uchar>(h+1,w+2));

				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//计算水平、竖直方向卷积插值算出的像素值
				//水平方向
				int p1=9.0/16.0*(tempMat.at<uchar>(h,w+1)+tempMat.at<uchar>(h,w-1))-1.0/16.0*(tempMat.at<uchar>(h,w+3)+tempMat.at<uchar>(h,w-3));

				//竖直方向
				int p2=9.0/16.0*(tempMat.at<uchar>(h-1,w)+tempMat.at<uchar>(h+1,w))-1.0/16.0*(tempMat.at<uchar>(h-3,w)+tempMat.at<uchar>(h+3,w));

				//自适应插值
				int p_adapt=(w1*p1+w2*p2)/(w1+w2);
				double T1=double(1+G1)/double(1+G2);
				double T2=double(1+G2)/double(1+G1);

				if (G1>G2)
				{
					if (T_value<T1&&T_value>T2)
						BGR[i].at<uchar>(h,w)=p2;
					else if(T_value<T2)
						BGR[i].at<uchar>(h,w)=p1;
					else
						BGR[i].at<uchar>(h,w)=p_adapt;
				}
				else if (G1<G2)
				{
					if (T_value<T2&&T_value>T1)
						BGR[i].at<uchar>(h,w)=p1;
					else if (T_value<T1)
						BGR[i].at<uchar>(h,w)=p2;
					else
						BGR[i].at<uchar>(h,w)=p_adapt;

				}
				else
					BGR[i].at<uchar>(h,w)=p_adapt;


				h+=2;
			}
			w+=2;
		}

	}
	
	Mat image_3;
	merge(BGR,3,image_3);
	imwrite("image_3.bmp",image_3);
	delete[] BGR;
	return 1;
}