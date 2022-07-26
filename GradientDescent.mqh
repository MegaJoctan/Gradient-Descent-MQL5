//+------------------------------------------------------------------+
//|                                              GradientDescent.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum costFx
  {
      MSE, //Mean Squared erro
      BCE, //binary cross entropy
      CUSTOM, //for debugging purposes
  };

enum Beta
  {
    Slope,
    Intercept
  };


#define DBL_MAX_MIN(val) if (val>DBL_MAX) Alert("Function ",__FUNCTION__,"\n Maximum Double value for ",#val," reached"); if (val<DBL_MIN && val>0) Alert("Function ",__FUNCTION__,"\n MInimum Double value for ",#val," reached") 

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CGradientDescent
  {
      private:
                           double e;
                           
                           double m_learning_rate;
                           int    m_iterations;
                           double m_XMatrix[];
                           double Y[];
                           int    m_rows;
                           int    m_cols;  
                           bool   m_debug;
      
      protected:
                           double Mse(double Bo, double Bi,Beta wrt);
                           double Bce(double Bo,double B1,Beta wrt);
                           
                           double CustomCostFunction(double x);
                           void   MatrixColumn(double &Matrix[],double &ColMatrix[],int column);
                           double Mean(double &data[]);
                           double std(double &data[]); //Standard deviation
                           double Sigmoid(double t);
      
      public:
                           CGradientDescent(int cols, double learning_rate=0.01,int iterations=1000, bool debugInfo=false);
                          ~CGradientDescent(void);
                          
                           void  GradientDescentFunction(double &XMatrix[], double &YMatrix[],costFx costFunction); 
                           void  MinMaxScaler(double &Array[]); 
                           void  ReadCsvCol(string filename,int col, double &Array[]);              
                          
  };
//+------------------------------------------------------------------+
CGradientDescent::CGradientDescent(int cols, double learning_rate=0.01,int iterations=1000, bool debugInfo=false)
 {
    e = 2.718281828; 
    
    m_learning_rate = learning_rate;
    m_cols = cols;
    m_iterations = iterations;
    m_debug = debugInfo;
    
 }
//+------------------------------------------------------------------+
CGradientDescent::~CGradientDescent(void)
 {
   ArrayFree(m_XMatrix);
   ArrayFree(Y);
 }
//+------------------------------------------------------------------+
void CGradientDescent::GradientDescentFunction(double &XMatrix[], double &YMatrix[],costFx costFunction)
 {
 
   ArrayCopy(m_XMatrix,XMatrix);
   ArrayCopy(Y,YMatrix);
   
   m_rows = ArraySize(Y);    
   
   if (m_cols*m_rows != ArraySize(Y)) Print("There is imbalance in number of rows in the Matrix of X and Y values");

//---

    Print("Gradient Descent CostFunction ",EnumToString(costFunction));

   double x0 = 0, x1=0;
   double b0=0, b1=0;
   int iters =0;
   
   if (costFunction == CUSTOM)
    {
       for (int i=0; i<m_iterations; i++)
        {
           x1 = x0 - m_learning_rate * CustomCostFunction(x0);
           
           if (m_debug)  printf("%d x0 = %.10f x1 = %.10f CostFunction = %.10f",iters,x0,x1,CustomCostFunction(x0)); 
           
           if (NormalizeDouble(CustomCostFunction(x0),8) == 0) { Print("Local miminum found =",x0);  break;  }
             
           x0 = x1;
        }
    }
         
//---
   
   double cost_B0=0, cost_B1=0;
   
   if (costFunction == MSE)
    {
      int iterations=0;
      for (int i=0; i<m_iterations; i++, iterations++)
        {
        
           cost_B0 = Mse(b0,b1,Intercept);
           cost_B1 = Mse(b0,b1,Slope);
           
           b0 = b0 - m_learning_rate * cost_B0; 
           b1 = b1 - m_learning_rate * cost_B1;
           
           if (m_debug) printf("%d b0 = %.8f cost_B0 = %.8f B1 = %.8f cost_B1 = %.8f",iterations,b0,cost_B0,b1,cost_B1);
           
           DBL_MAX_MIN(b0); DBL_MAX_MIN(cost_B0); DBL_MAX_MIN(cost_B1);
           
           if (NormalizeDouble(cost_B0,8) == 0 && NormalizeDouble(cost_B1,8) == 0)  break; 
           
        }
      printf("%d Iterations Local Minima are\nB0(Intercept) = %.5f  ||  B1(Coefficient) = %.5f",iterations,b0,b1);    
    }
//---
   if (costFunction == BCE)
    {
      int iterations=0;
      for (int i=0; i<m_iterations; i++, iterations++)
        {
        
           cost_B0 = Bce(b0,b1,Intercept);
           cost_B1 = Bce(b0,b1,Slope);
           
           b0 = b0 - m_learning_rate * cost_B0; 
           b1 = b1 - m_learning_rate * cost_B1;
           
           if (m_debug) printf("%d b0 = %.8f cost_B0 = %.8f B1 = %.8f cost_B1 = %.8f",iterations,b0,cost_B0,b1,cost_B1);
           
           DBL_MAX_MIN(b0); DBL_MAX_MIN(cost_B0); DBL_MAX_MIN(cost_B1);
           
           if (NormalizeDouble(cost_B0,8) == 0 && NormalizeDouble(cost_B1,8) == 0)  break; 
           
        }
      printf("%d Iterations Local Minima are\nB0(Intercept) = %.5f  ||  B1(Coefficient) = %.5f",iterations,b0,b1);    
    }
 }
//+------------------------------------------------------------------+
double CGradientDescent::CustomCostFunction(double x)
 {
   return(2 * ( x + 5 ));
 }
//+------------------------------------------------------------------+

double CGradientDescent::Mse(double Bo, double Bi, Beta wrt)
 {
   double sum_sqr=0;
   double m = ArraySize(Y);
   double x[];
   
   MatrixColumn(m_XMatrix,x,2);
   
//--- dcost wrt B0

   if (wrt == Intercept)
    {
      for (int i=0; i<ArraySize(Y); i++)
         {  
            //printf("y[%d] = %.5f  x[%d] = %.5f ",i,Y[i],i,x[i]);
            sum_sqr +=  Y[i] - (Bo + Bi*x[i]);
         }
       //Print("sum ",sum_sqr," cost ",(-2/m)*sum_sqr);  
       return((-2/m) * sum_sqr);
    } 

//--- dcost wrt Bi

   if (wrt == Slope)
    {
      for (int i=0; i<ArraySize(Y); i++)
         {  
            //printf("y[%d] = %.5f  x[%d] = %.5f ",i,Y[i],i,x[i]);
            sum_sqr += x[i] * (Y[i] - (Bo + Bi*x[i]));
         }
         
       return((-2/m) * sum_sqr);
    } 

    return(0);
 }
//+------------------------------------------------------------------+

double CGradientDescent::Bce(double Bo,double B1,Beta wrt)
 {
   double sum_sqr=0;
   double m = ArraySize(Y);
   double x[];
   
   MatrixColumn(m_XMatrix,x,2);
   
    if (wrt == Slope)
      for (int i=0; i<ArraySize(Y); i++)
        { 
          double Yp = Sigmoid(Bo+B1*x[i]);
          
          sum_sqr += (Y[i] - Yp) * x[i];
        }
        
    if (wrt == Intercept)
      for (int i=0; i<ArraySize(Y); i++)
         {
            double Yp = Sigmoid(Bo+B1*x[i]);
            sum_sqr += (Y[i] - Yp);
         } 
    return((-1/m)*sum_sqr);
 }
//+------------------------------------------------------------------+
double CGradientDescent::Sigmoid(double t)
 {
   return(1.0/(1+MathPow(e,-t)));
 }
//+------------------------------------------------------------------+
void CGradientDescent::MinMaxScaler(double &Array[])
 {
   double mean = Mean(Array);
   double max,min;
   double Norm[];
   
   ArrayResize(Norm,ArraySize(Array));
   
   max = Array[ArrayMaximum(Array)];   min = Array[ArrayMinimum(Array)];
   
    for (int i=0; i<ArraySize(Array); i++)
         Norm[i] = (Array[i] - min) / (max - min); 
   
   printf("Scaled data Mean = %.5f Std = %.5f",Mean(Norm),std(Norm));
   
   ArrayFree(Array);
   ArrayCopy(Array,Norm);
 }
 
//+------------------------------------------------------------------+

double CGradientDescent::Mean(double &data[])
 {
   double sum=0;
   
   for (int i=0; i<ArraySize(data); i++)
      sum += data[i]; // all values summation
           
    return(sum/ArraySize(data)); //total value after summation divided by total number of elements
 }
 
//+------------------------------------------------------------------+

double CGradientDescent::std(double &data[])
 {
   double mean =  Mean(data);
   double sum = 0;
   
    for (int i=0; i<ArraySize(data); i++)
       sum += MathPow(data[i] - mean,2);
    
    return(MathSqrt(sum/ArraySize(data)));  
 }
 
//+------------------------------------------------------------------+
void CGradientDescent::MatrixColumn(double &Matrix[],double &ColMatrix[],int column)
 {
   int cols = m_cols, rows = m_rows;
   int start = 0;
      
   ArrayResize(ColMatrix,m_rows);
   
    for (int k=0; k<cols+1; k++)
     {
      if (k+1 != column) continue;
      else
         {
           if (m_cols > 1) start = column-1;
           
            for (int i=0; i<rows; i++)
              {
                  ColMatrix[i] = Matrix[start];
                  
                  if (m_cols == 1)  start++;                   
                  else              start+=(rows-1);
              }
         }
     }  
 }
//+------------------------------------------------------------------+
void CGradientDescent::ReadCsvCol(string filename,int col,double &Array[])
 {
    int counter=0;
    
    int column = 0, rows=0;
    int from_column_number = col;
    
    int handle = FileOpen(filename,FILE_READ|FILE_CSV|FILE_ANSI,",",CP_UTF8);
    if (handle == INVALID_HANDLE) Print("Invalid csv handle Err =",GetLastError());
    else
    while (!FileIsEnding(handle))
      {
        double data = (double)FileReadString(handle);
        
        column++; 
//---      
        if (column==from_column_number)
           {
               if (rows>=1) //Avoid the first column which contains the column's header
                 {   
                     counter++;
                     ArrayResize(Array,counter); 
                     Array[counter-1]=data;
                 }   
                  
           }
         
//---
        if (FileIsLineEnding(handle))
          {                     
            rows++;
            column=0;
          }
      }
    FileClose(handle); 
 }
//+------------------------------------------------------------------+

