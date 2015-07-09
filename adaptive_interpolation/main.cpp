#include <opencv.hpp>
#include <math.h>

using namespace cv;

//ȫ�ֱ���
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

	//����ɫͼ��ֽ�Ϊ3��ͨ��
	Mat* BGR=new Mat[3];
	split(dst,BGR);



	for(int i=0;i<3;i++)
	{
		Mat tempMat=BGR[i];
		//��һ�ֲ�ֵ
		for (int w=6;w<2*W-6;)
		{
			for (int h=6;h<2*H-6;)
			{
				//�ֱ����45�ȣ�135�ȶԽ��߷����ݶ�
				//45��
				
				int G1=abs(tempMat.at<uchar>(h+3,w-3)-tempMat.at<uchar>(h+5,w-5))+abs(tempMat.at<uchar>(h+1,w-3)-tempMat.at<uchar>(h+3,w-5))+abs(tempMat.at<uchar>(h-1,w-3)-tempMat.at<uchar>(h+1,w-5))+abs(tempMat.at<uchar>(h-3,w-3)-tempMat.at<uchar>(h-1,w-5))+
					   abs(tempMat.at<uchar>(h+3,w-1)-tempMat.at<uchar>(h+5,w-3))+abs(tempMat.at<uchar>(h+1,w-1)-tempMat.at<uchar>(h+3,w-3))+abs(tempMat.at<uchar>(h-1,w-1)-tempMat.at<uchar>(h+1,w-3))+abs(tempMat.at<uchar>(h-3,w-1)-tempMat.at<uchar>(h-1,w-3))+
					   abs(tempMat.at<uchar>(h+3,w+1)-tempMat.at<uchar>(h+5,w-1))+abs(tempMat.at<uchar>(h+1,w+1)-tempMat.at<uchar>(h+3,w-1))+abs(tempMat.at<uchar>(h-1,w+1)-tempMat.at<uchar>(h+1,w-1))+abs(tempMat.at<uchar>(h-3,w+1)-tempMat.at<uchar>(h-1,w-1))+
					   abs(tempMat.at<uchar>(h+3,w+3)-tempMat.at<uchar>(h+5,w+1))+abs(tempMat.at<uchar>(h+1,w+3)-tempMat.at<uchar>(h+3,w+1))+abs(tempMat.at<uchar>(h-1,w+3)-tempMat.at<uchar>(h+1,w+1))+abs(tempMat.at<uchar>(h-3,w+3)-tempMat.at<uchar>(h-1,w+1));
				
				//135��
				int G2=abs(tempMat.at<uchar>(h-3,w-3)-tempMat.at<uchar>(h-5,w-5))+abs(tempMat.at<uchar>(h-1,w-3)-tempMat.at<uchar>(h-3,w-5))+abs(tempMat.at<uchar>(h+1,w-3)-tempMat.at<uchar>(h-1,w-5))+abs(tempMat.at<uchar>(h+3,w-3)-tempMat.at<uchar>(h+1,w-5))+
					   abs(tempMat.at<uchar>(h-3,w-1)-tempMat.at<uchar>(h-5,w-3))+abs(tempMat.at<uchar>(h-1,w-1)-tempMat.at<uchar>(h-3,w-3))+abs(tempMat.at<uchar>(h+1,w-1)-tempMat.at<uchar>(h-1,w-3))+abs(tempMat.at<uchar>(h+3,w-1)-tempMat.at<uchar>(h+1,w-3))+
					   abs(tempMat.at<uchar>(h-3,w+1)-tempMat.at<uchar>(h-5,w-1))+abs(tempMat.at<uchar>(h-1,w+1)-tempMat.at<uchar>(h-3,w-1))+abs(tempMat.at<uchar>(h+1,w+1)-tempMat.at<uchar>(h-1,w-1))+abs(tempMat.at<uchar>(h+3,w+1)-tempMat.at<uchar>(h+1,w-1))+
					   abs(tempMat.at<uchar>(h-3,w+3)-tempMat.at<uchar>(h-5,w+1))+abs(tempMat.at<uchar>(h-1,w+3)-tempMat.at<uchar>(h-3,w+1))+abs(tempMat.at<uchar>(h+1,w+3)-tempMat.at<uchar>(h-1,w+1))+abs(tempMat.at<uchar>(h+3,w+3)-tempMat.at<uchar>(h+1,w+1));

				
				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//����45�ȷ���135�ȷ������������ֵ���������ֵ
				//45��
				int p1=9.0/16.0*(tempMat.at<uchar>(h-1,w+1)+tempMat.at<uchar>(h+1,w-1))-1.0/16.0*(tempMat.at<uchar>(h-3,w+3)+tempMat.at<uchar>(h-3,w-3));
				
				//135��
				int p2=9.0/16.0*(tempMat.at<uchar>(h+1,w+1)+tempMat.at<uchar>(h-1,w-1))-1.0/16.0*(tempMat.at<uchar>(h+3,w+3)+tempMat.at<uchar>(h-3,w-3));

				p1=p1>255?255:p1;
				p2=p2>255?255:p2;

				//����Ӧ��ֵ
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

	//���еڶ��β�ֵ
	for (int i=0;i<3;i++)
	{
		Mat tempMat=BGR[i];
		for (int w=4;w<2*W-4;)
		{
			for(int h=3;h<2*H-4;)
			{
				//�ֱ����ˮƽ������ֱ�����ݶ�
				//ˮƽ�����ݶ�
				int G1=abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h-1,w+2))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h-1,w))+abs(tempMat.at<uchar>(h+1,w)-tempMat.at<uchar>(h+1,w+2))+abs(tempMat.at<uchar>(h+1,w-2)-tempMat.at<uchar>(h+1,w))+
				       abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h-2,w+1))+abs(tempMat.at<uchar>(h+2,w-1)-tempMat.at<uchar>(h+2,w+1));
				
				//��ֱ�����ݶ�
				int G2=abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h+2,w-1))+abs(tempMat.at<uchar>(h,w+1)-tempMat.at<uchar>(h+2,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h,w-1))+abs(tempMat.at<uchar>(h-2,w+1)-tempMat.at<uchar>(h,w+1))+
					   abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h+1,w))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h+1,w-2))+abs(tempMat.at<uchar>(h-1,w+2)-tempMat.at<uchar>(h+1,w+2));

				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//����ˮƽ����ֱ��������ֵ���������ֵ
				//ˮƽ����
				int p1=9.0/16.0*(tempMat.at<uchar>(h,w+1)+tempMat.at<uchar>(h,w-1))-1.0/16.0*(tempMat.at<uchar>(h,w+3)+tempMat.at<uchar>(h,w-3));

				//��ֱ����
				int p2=9.0/16.0*(tempMat.at<uchar>(h-1,w)+tempMat.at<uchar>(h+1,w))-1.0/16.0*(tempMat.at<uchar>(h-3,w)+tempMat.at<uchar>(h+3,w));

				//����Ӧ��ֵ
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
				//�ֱ����ˮƽ������ֱ�����ݶ�
				//ˮƽ�����ݶ�
				int G1=abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h-1,w+2))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h-1,w))+abs(tempMat.at<uchar>(h+1,w)-tempMat.at<uchar>(h+1,w+2))+abs(tempMat.at<uchar>(h+1,w-2)-tempMat.at<uchar>(h+1,w))+
					abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h-2,w+1))+abs(tempMat.at<uchar>(h+2,w-1)-tempMat.at<uchar>(h+2,w+1));

				//��ֱ�����ݶ�
				int G2=abs(tempMat.at<uchar>(h,w-1)-tempMat.at<uchar>(h+2,w-1))+abs(tempMat.at<uchar>(h,w+1)-tempMat.at<uchar>(h+2,w+1))+abs(tempMat.at<uchar>(h-2,w-1)-tempMat.at<uchar>(h,w-1))+abs(tempMat.at<uchar>(h-2,w+1)-tempMat.at<uchar>(h,w+1))+
					abs(tempMat.at<uchar>(h-1,w)-tempMat.at<uchar>(h+1,w))+abs(tempMat.at<uchar>(h-1,w-2)-tempMat.at<uchar>(h+1,w-2))+abs(tempMat.at<uchar>(h-1,w+2)-tempMat.at<uchar>(h+1,w+2));

				double w1=1.0/(1+pow(double(G1),K_value));
				double w2=1.0/(1+pow(double(G2),K_value));

				//����ˮƽ����ֱ��������ֵ���������ֵ
				//ˮƽ����
				int p1=9.0/16.0*(tempMat.at<uchar>(h,w+1)+tempMat.at<uchar>(h,w-1))-1.0/16.0*(tempMat.at<uchar>(h,w+3)+tempMat.at<uchar>(h,w-3));

				//��ֱ����
				int p2=9.0/16.0*(tempMat.at<uchar>(h-1,w)+tempMat.at<uchar>(h+1,w))-1.0/16.0*(tempMat.at<uchar>(h-3,w)+tempMat.at<uchar>(h+3,w));

				//����Ӧ��ֵ
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