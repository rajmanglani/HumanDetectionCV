/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package humandetectioncv;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;

/**
 *
 * @author rmanglani
 */
public class HumanDetectionCV {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
 
        Map<Double[], Double> train = new HashMap<>();
        File path = new File("C:\\Users\\RMANGLaNI\\Desktop\\Human\\Train_Positive");
        File[] files = path.listFiles();
        for(File f : files){
            int[][] imageMatrix = getGrayscaleMatrix(f);
            int[][] vert = verticalGradient(imageMatrix);
            int[][] horiz = horizontalGradient(imageMatrix);
            HOGDescriptor hog = new HOGDescriptor(edgeMagnitude(vert, horiz), edgeDegree(vert, horiz));
            List<Double> hogDesc  = hog.getHOGDesc();
            Double[] imageIn = new Double[hogDesc.size()];
            for(int i =0; i<imageIn.length; i++){
                imageIn[i] = hogDesc.get(i);
            }
            train.put(imageIn, 1.0);
        }
        
        File path2 = new File("C:\\Users\\RMANGLaNI\\Desktop\\Human\\Train_Negative");
        File[] files2 = path2.listFiles();
        for(File f : files2){
            int[][] imageMatrix = getGrayscaleMatrix(f);
            int[][] vert = verticalGradient(imageMatrix);
            int[][] horiz = horizontalGradient(imageMatrix);
            HOGDescriptor hog = new HOGDescriptor(edgeMagnitude(vert, horiz), edgeDegree(vert, horiz));
            List<Double> hogDesc  = hog.getHOGDesc();
            Double[] imageIn = new Double[hogDesc.size()];
            
            for(int i =0; i<imageIn.length; i++){
                imageIn[i] = hogDesc.get(i);
            }
            train.put(imageIn, 0.0);
        }
        
        NeuralNetwork nn = new NeuralNetwork(train, 0.001, 250,7524);
        nn.trainNetwork();

    }
    
    public static void printMat(Double[] mat){
        for(int i=0; i<mat.length; i++){
            System.out.print(mat[i] + " ");
        }
        System.out.println("");
    }
    
    public static int[][] getGrayscaleMatrix(File f) throws IOException{
        BufferedImage bi = ImageIO.read(f);
        int[][] imageMatrix = new int[bi.getWidth()][bi.getHeight()];
        for(int i =0; i<imageMatrix.length; i++){
            for(int j =0; j< imageMatrix[i].length; j++){
                int pixel = bi.getRGB(i, j);
                int red = (pixel >> 16) & 0xff;
                int green = (pixel >> 8) & 0xff;
                int blue = (pixel) & 0xff;
                imageMatrix[i][j] = Math.round((0.299f*red) + (0.587f*green) + (0.114f*blue));
            }
        }
        return imageMatrix;
    }
    public static void printMat(int[][] mat){
        for(int i =0; i<mat.length; i++){
            for(int j=0; j<mat[i].length; j++){
                System.out.print(mat[i][j] + " ");
            }
            System.out.println("----");
        }
    }
    
    /**
     * Calculates the vertical gradient of the smooth image using prewitts operator. 
     * @param img
     * @return 
     */
    public static int[][] verticalGradient(int[][] img){
        int[][] verticalMask = {{-1,0,1}, {-1,0,1}, {-1,0,1}}; //prewitt's operator vertical
        int[][] result = new int[img.length][img[0].length];
        
        //border of 1 px will be left since the operator mask will lie outside.
        int startI = verticalMask[0].length/2;
        int endI = img[0].length -1 -verticalMask[0].length/2;
        
        int startJ = verticalMask.length/2;
        int endJ = img.length - 1 - verticalMask.length/2;
        
        for(int i = startI; i<=endI; i++){
            for(int j = startJ; j<=endJ; j++){
                //perform convolution 
                int convValue = 0;
                for(int k =0; k<verticalMask.length; k++){
                    for(int l =0; l<verticalMask[k].length; l++){
                        convValue = convValue + (verticalMask[k][l] * (img[j-verticalMask.length/2+l][i-verticalMask[l].length/2+k]));
                    }
                }
                result[j][i] = Math.abs(convValue);  // put absolute value for normalization 0-255 pf pixel intensity values 
            }
        }
        return normalizeGradients(result); // normalize result 
    }
    
    /**
     * Calculates the horizontal gradient of the smooth image using prewitts operator. 
     * @param img
     * @return 
     */
    public static int[][] horizontalGradient(int[][] img){
        int[][] horizontalMask = {{-1,-1,-1}, {0,0,0}, {1,1,1}};  // prewitts operator horizontal
        int[][] result = new int[img.length][img[0].length];
        
        //border of 1 px will be left since the operator mask will lie outside.
        int startI = horizontalMask[0].length/2;
        int endI = img[0].length -1 -horizontalMask[0].length/2;
        
        int startJ = horizontalMask.length/2;
        int endJ = img.length - 1 - horizontalMask.length/2;
        
        for(int i = startI; i<=endI; i++){
            for(int j = startJ; j<=endJ; j++){
                //performs convolution 
                int convValue = 0;
                for(int k =0; k<horizontalMask.length; k++){
                    for(int l =0; l<horizontalMask[k].length; l++){
                        convValue = convValue + (horizontalMask[k][l] * (img[j-horizontalMask.length/2+l][i-horizontalMask[l].length/2+k]));
                    }
                }
                result[j][i] = Math.abs(convValue); // put absolute value for normalization 0-255 pf pixel intensity values 
            }
        }
         
        return normalizeGradients(result);
    }
    
    /**
     * Utility function to normalize the pixel values in a 2d image matrix between 0-255
     * @param img
     * @return 
     */
    public static int[][] normalizeGradients(int[][] img){
        //mapping 0-255
        //find highest 
        int highest = Integer.MIN_VALUE;
        for(int i =0; i<img.length; i++){
            for(int j =0; j<img[i].length; j++){
                if(img[i][j] > highest){
                    highest = img[i][j];
                }
            }
        }
        
        for(int i =0; i<img.length; i++){
            for(int j =0; j<img[i].length; j++){
                img[i][j] = Math.round(255*((float)img[i][j]/highest));   // map highest to 255, 0 will be lowest in our case. 
            }
        }
        return img;
    }
    
    /**
     * Calculates the edge magnitude given the vertical and horizontal gradients. x = sqrt(a^2+b^2)
     * @param vertical
     * @param horizontal
     * @return 
     */
    public static int[][] edgeMagnitude(int[][] vertical, int[][] horizontal){
        int[][] result = new int[vertical.length][vertical[0].length];
        for(int i=0; i<vertical.length; i++){
            for(int j =0; j<vertical[i].length; j++){
                result[i][j] = (int) Math.round(Math.sqrt(Math.pow(vertical[i][j], 2.0) + Math.pow(horizontal[i][j],2.0)));   // round to int value for ease of calculations
            }
        }
        return normalizeGradients(result);
    }
    
    /**
     * Calculates the edge degree for each pixel from the vertical and horizontal gradients 
     * @param vertical
     * @param horizontal
     * @return 
     */
    public static int[][] edgeDegree(int[][] vertical, int[][] horizontal){
        int[][] result = new int[vertical.length][vertical[0].length];
        for(int i=0; i<vertical.length; i++){
            for(int j =0; j<vertical[i].length; j++){
                // convert arc tan radians to degrees and round to int
                result[i][j] = (int) Math.round((Math.toDegrees(Math.atan2((double)vertical[i][j],(double)horizontal[i][j]))));  
            }
        }
        return result; 
    }
    
    
    
}
