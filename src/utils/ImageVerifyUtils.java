package utils;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import javax.imageio.ImageIO;

public class ImageVerifyUtils {

	BufferedImage image = null;
	//用來緩存圖片（在運行內存中）	
	Graphics2D gd = null;
	//畫筆的2D形式
	Random random = null; 
	//隨機數生成的類 
	StringBuffer charBuff = new StringBuffer(); 
	//存12345...90abc...xyzABC...XYZ，用來隨機取字符
	ArrayList<String> stringBuff = new ArrayList<String>(); 
	//字體列表，等下在方法中隨機取一個字體 
	ArrayList<Color> colorBuff = new ArrayList<Color>(); 
	//顏色列表，等下在方法中隨機取一個顏色 
	StringBuffer text = new StringBuffer();
	
	public ImageVerifyUtils(){ 
		random = new Random(); 
		image = new BufferedImage(160,40,BufferedImage.TYPE_INT_RGB); 
		gd = (Graphics2D)image.getGraphics(); 
		this.Init(); }

	private void Init() {
		charBuff.append("1234567890"); 
		charBuff.append("abcdefghigklmpqrstuvwxyz"); 
		charBuff.append("ABCDEFGHIGKLMPQRSTUVWXYZ"); 
		stringBuff.add("幼圓"); 
		stringBuff.add("宋體"); 
		stringBuff.add("華文琥珀"); 
		stringBuff.add("華文行楷"); 
		stringBuff.add("華文隸書"); 
	}

	/**
	*根據長、寬以及驗證碼的個數生成圖片驗證碼物件
	*@param width
	*@param height
	*@param randNo
	*@return
	* */
	public ImageVerifyItem drawImage(int width, int height, int randNo) {
		ImageVerifyItem item = new ImageVerifyItem();
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		gd = (Graphics2D)image.getGraphics(); 
		// 生成隨機類
		Random random = new Random();
		// 設定背景色
		gd.setColor(getRandColor(200, 250));
		// 填充指定的矩形
		gd.fillRect(0, 0, width, height);
		// 設定字型
		gd.setFont(new Font(getRandomFont(), Font.BOLD, image.getHeight()-20));
		int coe;
	
		
		
		// 隨機產生155條幹擾線,使影象中的認證碼不易被其他程式探測到
		for (int i = 0; i < getRandomNum(10,160); i++) {
			gd.setColor(getRandColor(160, 200));
			int x = random.nextInt(width);
			int y = random.nextInt(height);
			int xl = random.nextInt(12);
			int yl = random.nextInt(12);
			gd.drawLine(x, y, xl, yl);
		}
		
		gd.setColor(new Color(0,0,0));
		coe=getRandomNegative();
		int randwidth = (int)(coe*Math.random()*width*0.7);
		int randheight = getRandomNum((int)(height*0.75), (height/6));
		gd.drawOval(randwidth , randheight , (int)(width*1.7), (int)(height*1.5));
		
		gd .translate(0,image.getHeight()-15 );
		double randAngle = 0;
		int fontspace = (int) Math.cos(Math.toRadians(randAngle));
		// 取隨機產生的驗證碼
		for (int i = 0; i < randNo; i++) {
			char rand = getRandomChar();
			int fw;
			coe=getRandomNegative();
			text.append(rand);
			//獲取一個隨機字符，然後保存起來，以便之後獲取
			fw = (image.getWidth()-40)/randNo;
			gd .translate(getRandomNum(Math.max(fw,fontspace ), fw),0 );
			
			
			randAngle = coe*random.nextInt(90)*Math.PI/180;
			gd.rotate(randAngle);
			// 將驗證碼顯示到影象中
			gd.setColor(new Color(20 + random.nextInt(110), 20 + random.nextInt(110), 20 + random.nextInt(110)));
			gd.drawString(String.valueOf(rand),0, 0);
			gd.rotate(-randAngle);

		}
		// 釋放此圖形的上下文以及它使用的所有系統資源。
		gd.dispose();
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		try {
			ImageIO.write(image, "jpeg", baos);
		} catch (IOException e) {
			e.printStackTrace();
		}
		item.image = baos.toByteArray();
		return item;
	}
	/**
	* 根據給定範圍獲得隨機顏色
	*
	* */
	private Color getRandColor(int fc, int bc) {
		Random random = new Random();
		if (fc > 255)
		fc = 255;
		if (bc > 255)
		bc = 255;
		int r = fc + random.nextInt(bc - fc);
		int g = fc + random.nextInt(bc - fc);
		int b = fc + random.nextInt(bc - fc);
		return new Color(r, g, b);
	}
	private int getRandomNegative(){
		int coe;
		if (random.nextBoolean())
			coe=1;
		else
			coe=-1;
		return coe;
	}
	private int getRandomNum(int max,int min) {
		int rand = (int)Math.random()*(max-min)+min;		
		return rand;
		
	}
	private String getRandomFont()
	{ 
		int rand = random.nextInt(stringBuff.size()); 
		return stringBuff.get(rand); 
	}
	
	private  char getRandomChar()
	{ 
		int rand = random.nextInt(58); 
		return charBuff.charAt(rand); 
	}
	private  String getText() {
		System.out.println(text);
		return String.valueOf(text);
	}
	/**
	*	 輸出圖片到指定位址
	*
	* */
	public String output(String path,String num ,int w,int h,int n) {
		  // 產生四位數字的驗證碼
        ImageVerifyItem item =drawImage(w, h, n);
        String text = getText();
        String imgName = text;
        try {
        	//產生資料流
            FileOutputStream out = new FileOutputStream(path + "/" +num+".jpg" );
            out.write(item.image);
            if (out != null)
                out.close();
        } catch (IOException e ) {
            e.printStackTrace();
        }
        return imgName;
        
	} 
}
