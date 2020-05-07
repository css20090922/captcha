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
	//�Ψӽw�s�Ϥ��]�b�B�椺�s���^	
	Graphics2D gd = null;
	//�e����2D�Φ�
	Random random = null; 
	//�H���ƥͦ����� 
	StringBuffer charBuff = new StringBuffer(); 
	//�s12345...90abc...xyzABC...XYZ�A�Ψ��H�����r��
	ArrayList<String> stringBuff = new ArrayList<String>(); 
	//�r��C��A���U�b��k���H�����@�Ӧr�� 
	ArrayList<Color> colorBuff = new ArrayList<Color>(); 
	//�C��C��A���U�b��k���H�����@���C�� 
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
		stringBuff.add("����"); 
		stringBuff.add("����"); 
		stringBuff.add("�ؤ�[��"); 
		stringBuff.add("�ؤ�淢"); 
		stringBuff.add("�ؤ�����"); 
	}

	/**
	*�ھڪ��B�e�H�����ҽX���Ӽƥͦ��Ϥ����ҽX����
	*@param width
	*@param height
	*@param randNo
	*@return
	* */
	public ImageVerifyItem drawImage(int width, int height, int randNo) {
		ImageVerifyItem item = new ImageVerifyItem();
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		gd = (Graphics2D)image.getGraphics(); 
		// �ͦ��H����
		Random random = new Random();
		// �]�w�I����
		gd.setColor(getRandColor(200, 250));
		// ��R���w���x��
		gd.fillRect(0, 0, width, height);
		// �]�w�r��
		gd.setFont(new Font(getRandomFont(), Font.BOLD, image.getHeight()-20));
		int coe;
	
		
		
		// �H������155���F�Z�u,�ϼv�H�����{�ҽX�����Q��L�{��������
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
		// ���H�����ͪ����ҽX
		for (int i = 0; i < randNo; i++) {
			char rand = getRandomChar();
			int fw;
			coe=getRandomNegative();
			text.append(rand);
			//����@���H���r�šA�M��O�s�_�ӡA�H�K�������
			fw = (image.getWidth()-40)/randNo;
			gd .translate(getRandomNum(Math.max(fw,fontspace ), fw),0 );
			
			
			randAngle = coe*random.nextInt(90)*Math.PI/180;
			gd.rotate(randAngle);
			// �N���ҽX��ܨ�v�H��
			gd.setColor(new Color(20 + random.nextInt(110), 20 + random.nextInt(110), 20 + random.nextInt(110)));
			gd.drawString(String.valueOf(rand),0, 0);
			gd.rotate(-randAngle);

		}
		// ���񦹹ϧΪ��W�U��H�Υ��ϥΪ��Ҧ��t�θ귽�C
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
	* �ھڵ��w�d����o�H���C��
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
	*	 ��X�Ϥ�����w��}
	*
	* */
	public String output(String path,String num ,int w,int h,int n) {
		  // ���ͥ|��Ʀr�����ҽX
        ImageVerifyItem item =drawImage(w, h, n);
        String text = getText();
        String imgName = text;
        try {
        	//���͸�Ƭy
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
