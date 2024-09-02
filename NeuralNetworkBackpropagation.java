
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class NeuralNetworkBackpropagation {

    public static void main(String[] args) {
            String filename = "src/data/data_balita_1k.csv";//sesuaikan dengan alamat local file datasetnya
        String[] header = null;
        double[][] data = null;

        //----------------------------------------------------------------------
        // BACA FILE
        //----------------------------------------------------------------------
        File file = new File(filename);
        try {
            Scanner sc = new Scanner(file);

            // Baca header
            String baris = sc.nextLine();
            header = baris.split(",");
            // System.out.println(Arrays.toString(header));

            // Baca baris data
            ArrayList<String[]> dataString = new ArrayList<>();
            while (sc.hasNextLine()) {
                baris = sc.nextLine();
                String[] kolom = baris.split(",");
                if (kolom.length == header.length) {
                    dataString.add(kolom);
                }
            }

            // Transformasi data
            double[][] dataDouble = new double[dataString.size()][header.length];
            for (int i = 0; i < dataDouble.length; i++) {
                String[] kolom = dataString.get(i);

                // satisfaction
                String value = kolom[0];
                double umur = Double.parseDouble(value);
                dataDouble[i][0] = umur;

                // Jenis Kelamin
                value = kolom[1];
                double gender = 0;
                if (value.equalsIgnoreCase("perempuan")) {
                    gender = 0;
                } else if (value.equalsIgnoreCase("laki-laki")) {
                    gender = 1;
                }
                dataDouble[i][1] = gender;
                //tinggi badan
                value = kolom[2];
                double tinggi = Double.parseDouble(value);
                dataDouble[i][2] = tinggi;


                //status
                value = kolom[3];
                double status = 0;
                if (value.equalsIgnoreCase("severely stunted")) {
                    status = 0;
                } else if (value.equalsIgnoreCase("stunted")) {
                    status = 1;
                }else if (value.equalsIgnoreCase("normal")) {
                    status = 2;
                }else if (value.equalsIgnoreCase("tinggi")) {
                    status = 3;
                }
                dataDouble[i][3] = status;

            }

            // Min-Max
            double[] min = new double[header.length];
            double[] max = new double[header.length];
            for (int j = 0; j < header.length; j++) {
                min[j] = Double.MAX_VALUE;
                max[j] = Double.MIN_VALUE;
            }
            for (int i = 0; i < dataDouble.length; i++) {
                for (int j = 0; j < dataDouble[i].length; j++) {
                    double value = dataDouble[i][j];
                    if (value < min[j]) {
                        min[j] = value;
                    }
                    if (value > max[j]) {
                        max[j] = value;
                    }
                }
            }

            // Normalisasi
            data = new double[dataDouble.length][header.length];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    double value = dataDouble[i][j];
                    double normalValue = (value - min[j]) / (max[j] - min[j]);
                    data[i][j] = normalValue;
                }
            }

        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        }
        //----------------------------------------------------------------------
        // BACA FILE SELESAI
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // SPLIT DATA TRAINING DAN DATA TESTING
        //----------------------------------------------------------------------
        int numData = 0;
        int numDataTraining = 0;
        int numDataTesting = 0;
        double percentDataTraining = 70;
        double percentDataTesting = 30;
        double[][] dataTraining = null;
        double[][] dataTesting = null;
        if (header != null && data != null) {
            double totalPercent = percentDataTraining + percentDataTesting;
            numData = data.length;
            numDataTraining = (int) Math.floor(numData * (percentDataTraining / totalPercent));
            numDataTesting = (int) Math.floor(numData * (percentDataTesting / totalPercent));


            // SPLIT DATA TRAINING DAN DATA TESTING
            // set data training
            dataTraining = new double[numDataTraining][];
            for (int t = 0; t < dataTraining.length; t++) {
                dataTraining[t] = data[t];
            }
            // set data testing
            dataTesting = new double[numDataTesting][];
            for (int t = 0; t < dataTesting.length; t++) {
                dataTesting[t] = data[t + dataTraining.length];
            }
        }
        //----------------------------------------------------------------------
        // SPLIT DATA TRAINING DAN DATA TESTING SELESAI
        //---------------   -------------------------------------------------------

        //----------------------------------------------------------------------
        // PROSES TRAINING
        //----------------------------------------------------------------------
        int MAX_EPOCH = 150;
        int numInput = 1;
        int numHidden = 10;
        int numOutput = 1;
        double alpha = 0.01;
        double threshold = 0;
        double epsilonThreashold = 0.0000001;

        // Array Bobot atau model Network
        double[][] bobotV = null;
        double[][] bobotW = null;

        if (dataTraining != null) {
            // Persiapan Training
            numInput = dataTraining[0].length;
            double[][] V = new double[numInput][numHidden];
            double[][] W = new double[numHidden + 1][numOutput];

            // Ininisialisasi elemen-elemen array V dan W secara random
            Random r = new Random();
            // random array bobot V
            for (int i = 0; i < V.length; i++) {
                for (int j = 0; j < V[i].length; j++) {
                    V[i][j] = r.nextDouble();
                }
            }
            // random array bobot W
            for (int j = 0; j < W.length; j++) {
                for (int k = 0; k < W[j].length; k++) {
                    W[j][k] = r.nextDouble();
                }
            }

            boolean isConvergen = false;
            int epoch = 1;
            while (epoch <= MAX_EPOCH && !isConvergen) {
                int numYsamadenganT = 0;
                for (int t = 0; t < dataTraining.length; t++) {
                    // Inisialisasi Neuron Input dan Target
                    // T adalah target
                    // X adalah Neuron Input
                    numInput = dataTraining[t].length;
                    double[] T = new double[numOutput];
                    T[0] = dataTraining[t][0];
                    //geser satu index ke kiri untuk neuron input
                    double[] X = new double[numInput];
                    for (int i = 1; i < numInput; i++) {
                        X[i - 1] = dataTraining[t][i];
                    }
                    X[numInput - 1] = 1;// untuk bias

                    //----------------------------------------------------------
                    // feedforward
                    //----------------------------------------------------------
                    // Hitung Znet
                    double[] Znet = new double[numHidden];
                    for (int j = 0; j < numHidden; j++) {
                        double sum = 0;
                        for (int i = 0; i < numInput; i++) {
                            double XiVij = X[i] * V[i][j];
                            sum += XiVij;
                        }
                        Znet[j] = sum;
                    }

                    // Hitung Z
                    double[] Z = new double[numHidden + 1];
                    for (int j = 0; j < numHidden; j++) {
                        Z[j] = 1.0 / (1.0 + Math.exp(-Znet[j]));
                    }
                    Z[numHidden] = 1;// untuk bias

                    // Hitung Ynet
                    double[] Ynet = new double[numOutput];
                    for (int k = 0; k < numOutput; k++) {
                        double sum = 0;
                        for (int j = 0; j < Z.length; j++) {
                            double ZjWjk = Z[j] * W[j][k];
                            sum += ZjWjk;
                        }
                        Ynet[k] = sum;
                    }

                    // Hitung Y
                    double[] Y = new double[numOutput];
                    for (int k = 0; k < numOutput; k++) {
                        Y[k] = 1.0 / (1.0 + Math.exp(-Ynet[k]));
                    }

                    //update threshold
                    for (int k = 0; k < numOutput; k++) {
                        if (T[k] >= 1 && threshold > Y[k]) {
                            threshold = Y[k] - epsilonThreashold;
                        } else if (T[k] <= 0 && threshold < Y[k]) {
                            threshold = Y[k] + epsilonThreashold;
                        }
                    }

                    //----------------------------------------------------------
                    // backpropagation
                    //----------------------------------------------------------
                    int YT = 0;
                    // Hitung Faktor Error di layer output
                    double[] doY = new double[numOutput];//faktor do di unit output
                    for (int k = 0; k < numOutput; k++) {
                        doY[k] = (T[k] - Y[k]) * Y[k] * (1 - Y[k]);
                        //cek YsamadenganT
                        if (Y[k] == T[k]) {
                            YT++;
                        }
                    }

                    //cek numYsamadenganT
                    if (YT == T.length) {
                        numYsamadenganT++;
                    }

                    // Hitung deltaW
                    double[][] deltaW = new double[numHidden + 1][numOutput];
                    for (int j = 0; j < deltaW.length; j++) {
                        for (int k = 0; k < deltaW[j].length; k++) {
                            deltaW[j][k] = alpha * doY[k] * Z[j];
                        }
                    }

                    // Hitung doNetY
                    double[] doYnet = new double[numHidden];
                    for (int j = 0; j < numHidden; j++) {
                        double sum = 0;
                        for (int k = 0; k < numOutput; k++) {
                            double doYkWjk = doY[k] * W[j][k];
                            sum += doYkWjk;
                        }
                        doYnet[j] = sum;
                    }

                    // Hitung doZ
                    double[] doZ = new double[numHidden];
                    for (int j = 0; j < numHidden; j++) {
                        doZ[j] = doYnet[j] * Z[j] * (1 - Z[j]);
                    }

                    // Hitung deltaV
                    double[][] deltaV = new double[numInput][numHidden];
                    for (int i = 0; i < numInput; i++) {
                        for (int j = 0; j < numHidden; j++) {
                            deltaV[i][j] = alpha * doZ[j] * X[i];
                        }
                    }

                    //----------------------------------------------------------
                    // update bobot
                    //----------------------------------------------------------
                    // Update bobot W
                    for (int j = 0; j < W.length; j++) {
                        for (int k = 0; k < W[j].length; k++) {
                            W[j][k] = W[j][k] + deltaW[j][k];
                        }
                    }

                    // Update bobot V
                    for (int i = 0; i < V.length; i++) {
                        for (int j = 0; j < V[i].length; j++) {
                            V[i][j] = V[i][j] + deltaV[i][j];
                        }
                    }

                }//end fo for t

                // Cek Konvergensi
                if (numYsamadenganT == dataTraining.length) {
                    isConvergen = true;
                }

                //increment epoch
                epoch++;

            }//end of while(epoch <= MAX_EPOCH)

            // Set Bobot atau model Backpropagation Neural Network
            bobotV = V;
            bobotW = W;
        }
        //----------------------------------------------------------------------
        // PROSES TRAINING SELESAI
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // PROSES TESTING
        //----------------------------------------------------------------------
        int nTrue = 0;
        if (dataTesting != null && bobotV != null && bobotW != null) {
            for (int t = 0; t < dataTesting.length; t++) {
                numInput = dataTesting[t].length;
                double[] T = new double[numOutput];
                T[0] = dataTesting[t][0];
                //geser satu index ke kiri untuk neuron input
                double[] X = new double[numInput];
                for (int i = 1; i < numInput; i++) {
                    X[i - 1] = dataTesting[t][i];
                }
                X[numInput - 1] = 1;// untuk bias

                //----------------------------------------------------------
                // feedforward
                //----------------------------------------------------------
                // Hitung Znet
                double[] Znet = new double[numHidden];
                for (int j = 0; j < numHidden; j++) {
                    double sum = 0;
                    for (int i = 0; i < numInput; i++) {
                        double XiVij = X[i] * bobotV[i][j];
                        sum += XiVij;
                    }
                    Znet[j] = sum;
                }

                // Hitung Z
                double[] Z = new double[numHidden + 1];
                for (int j = 0; j < numHidden; j++) {
                    Z[j] = 1.0 / (1.0 + Math.exp(-Znet[j]));
                }
                Z[numHidden] = 1;// untuk bias

                // Hitung Ynet
                double[] Ynet = new double[numOutput];
                for (int k = 0; k < numOutput; k++) {
                    double sum = 0;
                    for (int j = 0; j < Z.length; j++) {
                        double ZjWjk = Z[j] * bobotW[j][k];
                        sum += ZjWjk;
                    }
                    Ynet[k] = sum;
                }

                // Hitung Y
                double[] Y = new double[numOutput];
                for (int k = 0; k < numOutput; k++) {
                    Y[k] = 1.0 / (1.0 + Math.exp(-Ynet[k]));
                }

                double[] Ytreshold = new double[numOutput];
                //double threshold = 0.91;
                for (int k = 0; k < numOutput; k++) {
                    if (Y[k] >=0 && Y[k]<=0.25) {
                        Ytreshold[k] = 0;
                    } else if(Y[k] >0.25 && Y[k]<=0.5) {
                        Ytreshold[k] = 1;
                    }else if(Y[k] >0.5 && Y[k]<=0.75) {
                        Ytreshold[k] = 2;
                    }
                    else if(Y[k] >0.75 && Y[k]<=1) {
                        Ytreshold[k] = 3;
                    }
                }

                //Trace Result
                System.out.print("data_testing-" + t + " Target - Prediksi ");
                for (int k = 0; k < numOutput; k++) {
                    if (k > 0) {
                        System.out.print(", ");
                    }
                    System.out.print("[" + T[k] + "-" + Ytreshold[k] + ":");
                    if (T[k] == Ytreshold[k]) {
                        System.out.print(" True ");
                        nTrue++;
                    } else {
                        System.out.print(" False ");
                    }
                    System.out.print("]");
                }
                System.out.println();

            }//end of for t

            double akurasi = ((double) nTrue / (double) numDataTesting) * 100;
            System.out.println("---------------------------------------------------");
            System.out.println("Threshold: "+threshold);
            System.out.println("AKURASI: " + String.format("%.2f", akurasi) + "%");
        }
        //----------------------------------------------------------------------
        // PROSES TESTING SELESAI
        //----------------------------------------------------------------------

    }//end of main
}