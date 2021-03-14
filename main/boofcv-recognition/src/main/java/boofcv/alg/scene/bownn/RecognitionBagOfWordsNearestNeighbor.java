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

package boofcv.alg.scene.bownn;

import org.ddogleg.nn.NearestNeighbor;
import org.ddogleg.struct.DogArray;

import java.util.List;

/**
 * Implementation of the "classical" Bog-Of-Words (BOW) (a.k.a. Bag-Of-Visual-Words) for object/scene recognition.
 * An image is described using a set of local image features (e.g. SIFT) which results in a set of n-dimensional
 * vectors. Each feature vector is converted into a word, which is then used to build a histogram of words in the
 * image. A similarity score is computed between two images using the histogram. Words are learned using k-means
 * clustering when applied to a large initial training set of image features.
 *
 * This implementation is designed to be simple and flexible. Allowing different algorithms in the same family
 * to be swapped out. For example, the nearest-neighbor (NN) search can be done using a brute force approach, kd-tree,
 * or an approximate kd-tree.
 *
 * This approach to image recognition appears to have many parents with no single paper/author being the primary
 * inspiration. One of the early works is cited below.
 * <p>
 *     <li>Sivic, Josef, and Andrew Zisserman. "Video Google: A text retrieval approach to object matching in videos."
 *     Computer Vision, IEEE International Conference on. Vol. 3. IEEE Computer Society, 2003.</li>
 * </p>
 *
 * @author Peter Abeles
 */
public class RecognitionBagOfWordsNearestNeighbor<Point> {
	NearestNeighbor<Point> searchNN;

	DogArray<Point> words;

	public void clearImages() {

	}

	public void addImage( int imageID, List<Point> imageFeatures ) {

	}

	public boolean query( List<Point> queryImage, int limit ) {
		return false;
	}
}
