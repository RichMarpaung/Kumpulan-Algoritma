import numpy
aktual = [83566.48,84429,   71787,   76276.33, 71194.79]
prediksi = [69232.19538462,67177.56423077, 65122.93307692, 63068.30192308, 61013.67076923]
    
sum = 0.0
for i in range (5) :
    sum +=(float) (aktual[i]-prediksi[i])**2
    
sum/=5
print(numpy.sqrt(sum))