//+------------------------------------------------------------------+
//|                                        gradient-descent test.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/job/new?prefered=omegajoctan"
#property version   "1.00"
//Hire me on your next big Machine learning project using the above link
#property description "Hire me on your next big Machine Learning project > click the above name"
#property script_show_inputs
//+------------------------------------------------------------------+
#include "GradientDescent.mqh";
CGradientDescent *grad;
//#include "C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\892B47EBC091D6EF95E3961284A76097\MQL5\Experts\DataScience\LinearRegression\LinearRegressionLib.mqh";
//CMatrixRegression *lr;
input bool PrintDebugInfo = true; //set this to false to avoid filling up the logs
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
      string filename = "Salary_Data.csv";
//---
      double XMatrix[];
      double YMatrix[];
      
      
      grad = new CGradientDescent(1, 0.1,10000,PrintDebugInfo);
      
      grad.ReadCsvCol(filename,1,XMatrix);
      grad.ReadCsvCol(filename,2,YMatrix);
       
      grad.MinMaxScaler(XMatrix);
      grad.MinMaxScaler(YMatrix);
      
//      
      //ArrayPrint("Normalized X",XMatrix);
      //ArrayPrint("Normalized Y",YMatrix);
      
      grad.GradientDescentFunction(XMatrix,YMatrix,MSE);
      
      filename = "titanic.csv";
      
      ZeroMemory(XMatrix);
      ZeroMemory(YMatrix);
      
      grad.ReadCsvCol(filename,3,XMatrix);
      grad.ReadCsvCol(filename,2,YMatrix);
      
      grad.GradientDescentFunction(XMatrix,YMatrix,BCE);
      
      delete (grad);
      
      //delete(lr);
  }
//+------------------------------------------------------------------+
void ArrayPrint(string str,double &Arr[])
 {
   Print(str);
   ::ArrayPrint(Arr);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+