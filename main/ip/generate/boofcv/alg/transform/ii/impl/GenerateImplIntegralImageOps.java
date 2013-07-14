/*
 * Copyright (c) 2011-2013, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.transform.ii.impl;

import boofcv.misc.AutoTypeImage;
import boofcv.misc.CodeGeneratorBase;
import boofcv.misc.CodeGeneratorUtil;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;


/**
 * @author Peter Abeles
 */
public class GenerateImplIntegralImageOps extends CodeGeneratorBase {
	String className = "ImplIntegralImageOps";

	PrintStream out;

	public GenerateImplIntegralImageOps() throws FileNotFoundException {
		out = new PrintStream(new FileOutputStream(className + ".java"));
	}

	@Override
	public void generate() throws FileNotFoundException {
		printPreamble();

		printTwoInput(AutoTypeImage.F32, AutoTypeImage.F32);
		printTwoInput(AutoTypeImage.F64, AutoTypeImage.F64);
		printTwoInput(AutoTypeImage.U8, AutoTypeImage.S32);
		printTwoInput(AutoTypeImage.S32, AutoTypeImage.S32);
		printTwoInput(AutoTypeImage.S64, AutoTypeImage.S64);
		singleInput(AutoTypeImage.F32);
		singleInput(AutoTypeImage.S32);
		singleInput(AutoTypeImage.F64);
		singleInput(AutoTypeImage.S64);

		out.print("\n" +
				"}\n");
	}

	private void printPreamble() {
		out.print(CodeGeneratorUtil.copyright);
		out.print("package boofcv.alg.transform.ii.impl;\n" +
				"\n" +
				"import boofcv.alg.transform.ii.IntegralKernel;\n" +
				"import boofcv.struct.ImageRectangle;\n" +
				"import boofcv.struct.image.*;\n" +
				"\n" +
				"\n" +
				"/**\n" +
				" * <p>\n" +
				" * Compute the integral image for different types of input images.\n" +
				" * </p>\n" +
				" * \n" +
				" * <p>\n" +
				" * DO NOT MODIFY: Generated by {@link GenerateImplIntegralImageOps}.\n" +
				" * </p>\n" +
				" * \n" +
				" * @author Peter Abeles\n" +
				" */\n" +
				"public class "+className+" {\n\n");
	}

	private void printTwoInput( AutoTypeImage imageIn , AutoTypeImage imageOut ) {
		printTransform(imageIn,imageOut);

	}

	private void singleInput(AutoTypeImage image) {
		printConvolve(image,image);
		printConvolveBorder(image,image);
		printConvolveSparse(image);
		printBlockUnsafe(image);
		printBlockZero(image);
	}

	private void printTransform( AutoTypeImage imageIn , AutoTypeImage imageOut ) {

		String sumType = imageOut.getSumType();
		String bitWise = imageIn.getBitWise();
		String typeCast = imageOut.getTypeCastFromSum();

		out.print("\tpublic static void transform( final "+imageIn.getImageName()+" input , final "+imageOut.getImageName()+" transformed )\n" +
				"\t{\n" +
				"\t\tint indexSrc = input.startIndex;\n" +
				"\t\tint indexDst = transformed.startIndex;\n" +
				"\t\tint end = indexSrc + input.width;\n" +
				"\n" +
				"\t\t"+sumType+" total = 0;\n" +
				"\t\tfor( ; indexSrc < end; indexSrc++ ) {\n" +
				"\t\t\ttransformed.data[indexDst++] = "+typeCast+"total += input.data[indexSrc]"+bitWise+";\n" +
				"\t\t}\n" +
				"\n" +
				"\t\tfor( int y = 1; y < input.height; y++ ) {\n" +
				"\t\t\tindexSrc = input.startIndex + input.stride*y;\n" +
				"\t\t\tindexDst = transformed.startIndex + transformed.stride*y;\n" +
				"\t\t\tint indexPrev = indexDst - transformed.stride;\n" +
				"\n" +
				"\t\t\tend = indexSrc + input.width;\n" +
				"\n" +
				"\t\t\ttotal = 0;\n" +
				"\t\t\tfor( ; indexSrc < end; indexSrc++ ) {\n" +
				"\t\t\t\ttotal +=  input.data[indexSrc]"+bitWise+";\n" +
				"\t\t\t\ttransformed.data[indexDst++] = transformed.data[indexPrev++] + total;\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printConvolve( AutoTypeImage imageIn , AutoTypeImage imageOut) {
		out.print("\tpublic static void convolve( "+imageIn.getImageName()+" integral ,\n" +
				"\t\t\t\t\t\t\t\t ImageRectangle[] blocks , int scales[],\n" +
				"\t\t\t\t\t\t\t\t "+imageOut.getImageName()+" output )\n" +
				"\t{\n" +
				"\t\tfor( int y = 0; y < integral.height; y++ ) {\n" +
				"\t\t\tfor( int x = 0; x < integral.width; x++ ) {\n" +
				"\t\t\t\t"+imageIn.getSumType()+" total = 0;\n" +
				"\t\t\t\tfor( int i = 0; i < blocks.length; i++ ) {\n" +
				"\t\t\t\t\tImageRectangle b = blocks[i];\n" +
				"\t\t\t\t\ttotal += block_zero(integral,x+b.x0,y+b.y0,x+b.x1,y+b.y1)*scales[i];\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y,total);\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printConvolveBorder(AutoTypeImage imageIn , AutoTypeImage imageOut) {
		String sumType = imageIn.getSumType();
		out.print("\tpublic static void convolveBorder( "+imageIn.getImageName()+" integral ,\n" +
				"\t\t\t\t\t\t\t\t\t   ImageRectangle[] blocks , int scales[],\n" +
				"\t\t\t\t\t\t\t\t\t   "+imageOut.getImageName()+" output , int borderX , int borderY )\n" +
				"\t{\n" +
				"\t\tfor( int x = 0; x < integral.width; x++ ) {\n" +
				"\t\t\tfor( int y = 0; y < borderY; y++ ) {\n" +
				"\t\t\t\t"+sumType+" total = 0;\n" +
				"\t\t\t\tfor( int i = 0; i < blocks.length; i++ ) {\n" +
				"\t\t\t\t\tImageRectangle b = blocks[i];\n" +
				"\t\t\t\t\ttotal += block_zero(integral,x+b.x0,y+b.y0,x+b.x1,y+b.y1)*scales[i];\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y,total);\n" +
				"\t\t\t}\n" +
				"\t\t\tfor( int y = integral.height-borderY; y < integral.height; y++ ) {\n" +
				"\t\t\t\t"+sumType+" total = 0;\n" +
				"\t\t\t\tfor( int i = 0; i < blocks.length; i++ ) {\n" +
				"\t\t\t\t\tImageRectangle b = blocks[i];\n" +
				"\t\t\t\t\ttotal += block_zero(integral,x+b.x0,y+b.y0,x+b.x1,y+b.y1)*scales[i];\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y,total);\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\n" +
				"\t\tint endY = integral.height-borderY;\n" +
				"\t\tfor( int y = borderY; y < endY; y++ ) {\n" +
				"\t\t\tfor( int x = 0; x < borderX; x++ ) {\n" +
				"\t\t\t\t"+sumType+" total = 0;\n" +
				"\t\t\t\tfor( int i = 0; i < blocks.length; i++ ) {\n" +
				"\t\t\t\t\tImageRectangle b = blocks[i];\n" +
				"\t\t\t\t\ttotal += block_zero(integral,x+b.x0,y+b.y0,x+b.x1,y+b.y1)*scales[i];\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y,total);\n" +
				"\t\t\t}\n" +
				"\t\t\tfor( int x = integral.width-borderX; x < integral.width; x++ ) {\n" +
				"\t\t\t\t"+sumType+" total = 0;\n" +
				"\t\t\t\tfor( int i = 0; i < blocks.length; i++ ) {\n" +
				"\t\t\t\t\tImageRectangle b = blocks[i];\n" +
				"\t\t\t\t\ttotal += block_zero(integral,x+b.x0,y+b.y0,x+b.x1,y+b.y1)*scales[i];\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y,total);\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printConvolveSparse(AutoTypeImage image ) {
		String sumType = image.getSumType();

		out.print("\tpublic static "+sumType+" convolveSparse( "+image.getImageName()+" integral , IntegralKernel kernel , int x , int y )\n" +
				"\t{\n" +
				"\t\t"+sumType+" ret = 0;\n" +
				"\t\tint N = kernel.getNumBlocks();\n" +
				"\n" +
				"\t\tfor( int i = 0; i < N; i++ ) {\n" +
				"\t\t\tImageRectangle r = kernel.blocks[i];\n" +
				"\t\t\tret += block_zero(integral,x+r.x0,y+r.y0,x+r.x1,y+r.y1)*kernel.scales[i];\n" +
				"\t\t}\n" +
				"\n" +
				"\t\treturn ret;\n" +
				"\t}\n\n");
	}

	private void printBlockUnsafe( AutoTypeImage image ) {
		String sumType = image.getSumType();
		String bitWise = image.getBitWise();

		out.print("\tpublic static "+sumType+" block_unsafe( "+image.getImageName()+" integral , int x0 , int y0 , int x1 , int y1 )\n" +
				"\t{\n" +
				"\t\t"+sumType+" br = integral.data[ integral.startIndex + y1*integral.stride + x1 ]"+bitWise+";\n" +
				"\t\t"+sumType+" tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ]"+bitWise+";\n" +
				"\t\t"+sumType+" bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ]"+bitWise+";\n" +
				"\t\t"+sumType+" tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ]"+bitWise+";\n" +
				"\n" +
				"\t\treturn br-tr-bl+tl;\n" +
				"\t}\n\n");
	}

	private void printBlockZero( AutoTypeImage image ) {
		String sumType = image.getSumType();
		String bitWise = image.getBitWise();

		out.print("\tpublic static "+sumType+" block_zero( "+image.getImageName()+" integral , int x0 , int y0 , int x1 , int y1 )\n" +
				"\t{\n" +
				"\t\tx0 = Math.min(x0,integral.width-1);\n" +
				"\t\ty0 = Math.min(y0,integral.height-1);\n" +
				"\t\tx1 = Math.min(x1,integral.width-1);\n" +
				"\t\ty1 = Math.min(y1,integral.height-1);\n" +
				"\n" +
				"\t\t"+sumType+" br=0,tr=0,bl=0,tl=0;\n" +
				"\n" +
				"\t\tif( x1 >= 0 && y1 >= 0)\n" +
				"\t\t\tbr = integral.data[ integral.startIndex + y1*integral.stride + x1 ]"+bitWise+";\n" +
				"\t\tif( y0 >= 0 && x1 >= 0)\n" +
				"\t\t\ttr = integral.data[ integral.startIndex + y0*integral.stride + x1 ]"+bitWise+";\n" +
				"\t\tif( x0 >= 0 && y1 >= 0)\n" +
				"\t\t\tbl = integral.data[ integral.startIndex + y1*integral.stride + x0 ]"+bitWise+";\n" +
				"\t\tif( x0 >= 0 && y0 >= 0)\n" +
				"\t\t\ttl = integral.data[ integral.startIndex + y0*integral.stride + x0 ]"+bitWise+";\n" +
				"\n" +
				"\t\treturn br-tr-bl+tl;\n" +
				"\t}\n\n");
	}

	public static void main( String args[] ) throws FileNotFoundException {
		GenerateImplIntegralImageOps app = new GenerateImplIntegralImageOps();
		app.generate();
	}
}
