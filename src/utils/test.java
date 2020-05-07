package utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;

import org.junit.jupiter.api.Test;

public class test {
@Test
	public void test() {
	int times=0;
	String  label,path,folder;
	String usage = "";
	File file;
	
	path= System.getProperty("user.dir"); 
	
//	//���ͰV�m�θ��
//	usage = "train";
//	folder = path+"\\"+usage+"img";
//	file =new File(folder); 
//	//�T�{�O�_�s�b
//	CheckDir(file);
//	
//	times=50000;
//	label = outputimg(folder,times);
//	outputtxt(usage,label);
//	
	//���ʹ��եθ��
	usage="test";
	folder = path+"\\"+usage+"img";
	file =new File(folder); 
	//�T�{�O�_�s�b
	CheckDir(file);
	
	times=10000;
	label = outputimg(folder,times);
	outputtxt(usage,label);
	}


	public String outputimg(String path,int times) {
		StringBuffer ans = new StringBuffer();
		String  label;
		
		for(int i=0;i<times;i++) {
			ImageVerifyUtils imgutils = new  ImageVerifyUtils();
			String num = String.format("%04d", i);
			int fnum = (int) (Math.random()*2+4);
			System.out.println(i+":"+path);
			label = imgutils.output(path,num,150,60,fnum);
			ans.append(label+",");
			
		}
		//�Nstringbuffer�নstring
		 label = String.valueOf(ans);
		 label=label.substring(0,label.length()-1);
		return label;
		
	}
	public void CheckDir(File targetFile) {
	if  (!targetFile .exists()  && !targetFile .isDirectory())      
	{       
	    System.out.println("//���s�b");  
	    targetFile .mkdir();    
	} else   
	{  
	    System.out.println("//�ؿ��s�b");  
	}  
	}
	public void outputtxt(String usage,String label) {
		try {
			System.out.println("start output "+usage);
			File output = new File(usage+"label.txt");
			if (!output.exists()) { // ���s�b�ɮ׮ɡA�إ�
			    output.createNewFile();
			} else { // �ɮצs�b�ɧR��
				output.delete();
			}
			BufferedWriter out = new BufferedWriter(new FileWriter(output));
	        out.write(label);
	        out.flush();
	        if (out != null) {
	        	//�����ɮ�
	        	out.close();
	        	System.out.println(usage+"output end ");
	        }
	        
	    } catch (IOException e ) {
	        e.printStackTrace();
	    }
	}
}
