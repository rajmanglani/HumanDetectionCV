/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package humandetectioncv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author rmanglani
 */
public class HOGDescriptor {
    
    private int[][] edgeMag;
    private int[][] edgeDeg;
    private List<double[]> histograms;

    public HOGDescriptor(int[][] edgeMag, int[][] edgeDeg) {
        this.edgeMag = edgeMag;
        this.edgeDeg = edgeDeg;
        histograms = new ArrayList<>();
        generateBlockDescriptor();
    }
    
    public List<Double> getHOGDesc(){
        List<Double> desc = new ArrayList<>();
        for(double[] arr : histograms){
            for(int i=0; i<arr.length;i++){
                desc.add(arr[i]);
            }
        }
        return desc;
    }
    
    private void generateBlockDescriptor(){
        //generate histogram for a block -- size 36
        
        Map<Integer,double[]> cellHistograms = new HashMap<>();   // maps cell id to its histogram 
        int id = 0;
        for(int i = 0; i<12; i++){
            for(int j = 0; j<20; j++){
                //for each cell 
                int[][] cellMag = new int[8][8];
                int[][] cellDeg = new int[8][8];
                int cellStartI = i*8; 
                for(int k=0; k<8; k++){
                    int cellStartJ = j*8;
                    for(int m=0; m<8; m++){
                        cellMag[k][m] = edgeMag[cellStartI][cellStartJ];
                        cellDeg[k][m] = edgeDeg[cellStartI][cellStartJ];
                        cellStartJ++;
                    }
                    cellStartI++;
                }
                cellHistograms.put(id, genHistogramForCell(cellMag, cellDeg));
                id++;
            }
        }
        
//        for(Map.Entry<Integer, double[]> e : cellHistograms.entrySet()){
//            System.out.println(e.getKey());
//            for(int i =0; i<e.getValue().length; i++){
//                System.out.print(e.getValue()[i]+ " ");
//            }
//            System.out.println("");
//        }
        
        for(int i=0; i<11; i++){
            for(int j=0; j<19;j++){
                //for every block get the 4 cells and normalize the vectors
                int cell1 = i*20 + j;
                int cell2 = cell1+1;
                int cell3 = cell1+20;
                int cell4 = cell3+1;
                histograms.add(normalizeVectors(cellHistograms.get(cell1), cellHistograms.get(cell2), cellHistograms.get(cell3), cellHistograms.get(cell4)));
                
            }
        }
        
    }
    
    private double[] normalizeVectors(double[] v1, double[] v2, double[] v3, double[] v4){
        //concat and form a new vector  - 4 vectors of length 9 
        double[] res = new double[36];
        int index = 0;
        for(int i=0; i<v1.length; i++){
            res[index] = v1[i];
            index++;
        }
        for(int i=0; i<v2.length; i++){
            res[index] = v2[i];
            index++;
        }
        for(int i=0; i<v3.length; i++){
            res[index] = v3[i];
            index++;
        }
        for(int i=0; i<v4.length; i++){
            res[index] = v4[i];
            index++;
        }
        
        double magSum = 0.0;
        for(int i =0; i<res.length;i++){
            magSum += Math.pow(res[i], 2.0);
        }
        double l2Norm = Math.sqrt(magSum);
        for(int i=0; i<res.length; i++){
            res[i] = res[i]/l2Norm;
        }
        return res;
    }
    
    private double[] genHistogramForCell(int[][] cellMag, int[][] cellEdge){
        double[] histogram = new double[9];  // 9 bins to add to 
        for(int i =0; i<cellMag.length; i++){
            for(int j =0; j<cellMag[i].length; j++){
                int edgeDeg = cellEdge[i][j];
                int edgeMag = cellMag[i][j];
                int binNum = 0;
                if(edgeDeg >= 170 && edgeDeg < 350){
                    edgeDeg = edgeDeg - 180;
                }
                
                //get the bin for the pixel
                int bin = getBin(edgeDeg);
                if(edgeDeg == (bin*20)){
                    //if its the center add to that bin 
                    histogram[bin] += edgeMag;
                }else if(edgeDeg > (bin*20)){
                    //more than center then part goes in this bin and rest in the next bin 
                    double x = ((edgeDeg - (bin*20))/20.0)*edgeMag;  
                    histogram[bin] += (edgeMag-x);
                    //for the next bin
                    histogram[(bin+1)%9] += x;
                }else{
                    //less than center then part goes in this bin and part in the previous bin 
                    double x = (((bin*20) - edgeDeg)/20.0)*edgeMag;  
                    histogram[bin] += (edgeMag-x);
                    //for the previous bin
                    histogram[(bin-1)%9] += x;
                }
            }
        }
        
        return histogram;
    }
    
    private int getBin(int deg){
        if(deg >= 10 && deg < 30){
            return 1;
        }else if(deg >= 30 && deg < 50){
            return 2;
        }else if(deg >= 50 && deg < 70){
            return 3;
        }else if(deg >= 70 && deg < 90){
            return 4;
        }else if(deg >= 90 && deg < 110){
            return 5;
        }else if(deg >= 110 && deg < 130){
            return 6;
        }else if(deg >= 130 && deg < 150){
            return 7;
        }else if(deg >= 150 && deg < 170){
            return 8;
        }else{
            return 0;
        }
    }
    
    
}
