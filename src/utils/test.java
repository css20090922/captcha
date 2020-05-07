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
	
//	//產生訓練用資料
//	usage = "train";
//	folder = path+"\\"+usage+"img";
//	file =new File(folder); 
//	//確認是否存在
//	CheckDir(file);
//	
//	times=50000;
//	label = outputimg(folder,times);
//	outputtxt(usage,label);
//	
	//產生測試用資料
	usage="test";
	folder = path+"\\"+usage+"img";
	file =new File(folder); 
	//確認是否存在
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
		//將stringbuffer轉成string
		 label = String.valueOf(ans);
		 label=label.substring(0,label.length()-1);
		return label;
		
	}
	public void CheckDir(File targetFile) {
	if  (!targetFile .exists()  && !targetFile .isDirectory())      
	{       
	    System.out.println("//不存在");  
	    targetFile .mkdir();    
	} else   
	{  
	    System.out.println("//目錄存在");  
	}  
	}
	public void outputtxt(String usage,String label) {
		try {
			System.out.println("start output "+usage);
			File output = new File(usage+"label.txt");
			if (!output.exists()) { // 不存在檔案時，建立
			    output.createNewFile();
			} else { // 檔案存在時刪除
				output.delete();
			}
			BufferedWriter out = new BufferedWriter(new FileWriter(output));
	        out.write(label);
	        out.flush();
	        if (out != null) {
	        	//關閉檔案
	        	out.close();
	        	System.out.println(usage+"output end ");
	        }
	        
	    } catch (IOException e ) {
	        e.printStackTrace();
	    }
	}
}
