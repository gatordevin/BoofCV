/*
 * Copyright (c) 2021, Peter Abeles. All Rights Reserved.
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

package boofcv.abst.scene.nn;

import boofcv.abst.scene.ImageRecognition;
import boofcv.struct.feature.TupleDesc;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageType;
import org.ddogleg.struct.DogArray;
import org.jetbrains.annotations.Nullable;

import java.io.PrintStream;
import java.util.Iterator;
import java.util.Set;

/**
 * Image recognition based on {@link boofcv.alg.scene.nn.RecognitionNearestNeighborInvertedFile}.
 *
 * @author Peter Abeles
 */
public class ImageRecognitionNearestNeighborInvertedFile<Image extends ImageBase<Image>, TD extends TupleDesc<TD>>
		implements ImageRecognition<Image> {
	@Override public void learnModel( Iterator<Image> images ) {

	}

	@Override public void clearDatabase() {

	}

	@Override public void addImage( String id, Image image ) {

	}

	@Override public boolean query( Image queryImage, int limit, DogArray<Match> matches ) {
		return false;
	}

	@Override public ImageType<Image> getImageType() {
		return null;
	}

	@Override public void setVerbose( @Nullable PrintStream printStream, @Nullable Set<String> set ) {

	}
}
