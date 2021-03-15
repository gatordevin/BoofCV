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

package boofcv.alg.scene.nn;

import boofcv.alg.scene.nister2006.TupleMapDistanceNorm;
import boofcv.alg.scene.nister2006.TupleMapDistanceNorm.CommonWords;
import lombok.Getter;
import lombok.Setter;
import org.ddogleg.nn.NearestNeighbor;
import org.ddogleg.nn.NnData;
import org.ddogleg.struct.*;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
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
 * The image database is stored using an inverted file for quick retrieval and error computation, as was done in [2].
 *
 * TODO go over specifics
 *
 * There is no single source for this specific implementation and borrows ideas from several papers. The paper
 * below is one of the earlier works to discuss the concept for visual BOW.
 * <ol>
 * <li>Sivic, Josef, and Andrew Zisserman. "Video Google: A text retrieval approach to object matching in videos."
 * Computer Vision, IEEE International Conference on. Vol. 3. IEEE Computer Society, 2003.</li>
 * <li>Nister, David, and Henrik Stewenius. "Scalable recognition with a vocabulary tree."
 * 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06). Vol. 2. Ieee, 2006.</li>
 * </ol>
 *
 * @author Peter Abeles
 */
public class RecognitionNearestNeighborInvertedFile<Point> {
	/** A nearest-neighbor search to look up the closest fit to each word */
	protected @Getter NearestNeighbor<Point> searchNN;

	/** Distance between two TF-IDF descriptors. L1 and L2 norms are provided */
	protected @Getter @Setter TupleMapDistanceNorm distanceFunction = new TupleMapDistanceNorm.L2();

	/** List of images added to the database */
	protected @Getter final BigDogArray_I32 imagesDB = new BigDogArray_I32(100, 10000, BigDogArray.Growth.GROW_FIRST);

	/** List of all possible images it could match with */
	@Getter DogArray<Candidate> matches = new DogArray<>(Candidate::new, Candidate::reset);

	/** List of images in the DB that are observed by each word. One element per word. */
	@Getter DogArray<InvertedFile> invertedFiles = new DogArray<>(InvertedFile::new, InvertedFile::reset);

	// Look up table from image to match. All values but be set to -1 after use
	// The size of this array will be the same as the number of DB images
	DogArray_I32 image_to_matches = new DogArray_I32();

	// Histogram for the number of times each word appears. All values must be 0 initially
	// One element for each word
	DogArray_I32 wordHistogram = new DogArray_I32();
	// List of words which were observed
	DogArray_I32 observedWords = new DogArray_I32();

	// temporary storage for an image TF-IDF descriptor
	DogArray_F32 tmpDescWeights = new DogArray_F32();

	/**
	 * Initializes the data structures.
	 *
	 * @param searchNN Search used to find the words.
	 * @param numWords Number of words
	 */
	public void initialize( NearestNeighbor<Point> searchNN, int numWords ) {
		this.searchNN = searchNN;
		invertedFiles.resize(numWords);
		imagesDB.reset();
	}

	/**
	 * Discards all memory of words which were added
	 */
	public void clearImages() {
		imagesDB.reset();

		// Clear the inverted files list. This will force all elements to be reset
		int numWords = invertedFiles.size;
		invertedFiles.reset();
		invertedFiles.resize(numWords);
	}

	/**
	 * Adds a new image to the database.
	 *
	 * @param imageID The image's unique ID for later reference
	 * @param imageFeatures Feature descriptors from an image
	 */
	public void addImage( int imageID, List<Point> imageFeatures ) {
		int imageIdx = imagesDB.size;
		imagesDB.append(imageID);

		computeWordHistogram(imageFeatures);
		computeImageDescriptor(imageFeatures.size());

		// Add this image to the inverted file for each word
		for (int i = 0; i < observedWords.size; i++) {
			int word = observedWords.get(i);
			invertedFiles.get(word).addImage(imageIdx, tmpDescWeights.get(i));
		}
	}

	/**
	 * Computes the number of times each word appears in the list of features
	 */
	private void computeWordHistogram( List<Point> imageFeatures ) {
		// Find and count the number of times each word appears in this set of features
		NearestNeighbor.Search<Point> search = searchNN.createSearch();
		NnData<Point> result = new NnData<>();
		observedWords.reset();
		for (int featureIdx = 0; featureIdx < imageFeatures.size(); featureIdx++) {
			if (!search.findNearest(imageFeatures.get(featureIdx), -1, result))
				continue;

			int count = wordHistogram.data[result.index];
			wordHistogram.data[result.index]++;
			if (count == 0) {
				observedWords.add(result.index);
			}
		}
	}

	/**
	 * Given the image histogram, compute the TF-IDF descriptor
	 *
	 * @param totalImageFeatures Number of features in this image
	 */
	private void computeImageDescriptor( float totalImageFeatures ) {
		// Compute the weight for each word in the descriptor based on its frequency
		tmpDescWeights.reset();
		for (int i = 0; i < observedWords.size; i++) {
			int word = observedWords.get(i);
			float weight = wordHistogram.get(word)/totalImageFeatures;
			tmpDescWeights.add(weight);

			// make sure the histogram is full of zeros again
			wordHistogram.set(word, 0);
		}

		// Normalize the image descriptor
		distanceFunction.normalize(tmpDescWeights);
	}

	/**
	 * Looks up the best match from the database. The list of all potential matches can be accessed by calling
	 * {@link #getMatches()}.
	 *
	 * @param queryImage Set of feature descriptors from the query image
	 * @param limit Maximum number of matches it will return.
	 * @return The best matching image with score from the database
	 */
	public boolean query( List<Point> queryImage, int limit ) {
		computeWordHistogram(queryImage);
		computeImageDescriptor(queryImage.size());
		findMatches();

		// Compute the score for each candidate and other book keeping
		for (int candidateIter = 0; candidateIter < matches.size; candidateIter++) {
			Candidate c = matches.get(candidateIter);
			c.error = distanceFunction.distance(c.commonWords.toList());

			// Ensure this array is once again full of -1
			image_to_matches.set(c.identification, -1);

			// convert it from image index into the user provided ID number
			c.identification = imagesDB.get(c.identification);
		}

		// Select the best candidate
		Collections.sort(matches.toList());
		// reduce it to being the k-best matches
		matches.size = Math.min(limit, matches.size);

		return matches.size > 0;
	}

	/**
	 * Finds all the matches using the observed words and the inverted files.
	 */
	private void findMatches() {
		// This will always be filled with -1 initially
		image_to_matches.resize(imagesDB.size, -1);

		// Create a list of all candidate images in the DB
		matches.reset();
		for (int wordIdx = 0; wordIdx < observedWords.size; wordIdx++) {
			int word = observedWords.get(wordIdx);
			InvertedFile invertedFile = invertedFiles.get(word);

			// Go through the inverted file list
			final int N = invertedFile.wordWeights.size;
			for (int invertedIdx = 0; invertedIdx < N; invertedIdx++) {
				int imageIndex = invertedFile.images.get(invertedIdx);

				// See if this DB image has been seen before
				Candidate c;
				int matchIdx = image_to_matches.get(imageIndex);
				if (matchIdx == -1) {
					// It has not been seen before, create a new entry for it in the candidate list
					matchIdx = matches.size;
					image_to_matches.set(imageIndex, matchIdx);
					c = matches.grow();
				} else {
					c = matches.get(matchIdx);
				}

				// Store the weights from each descriptor
				c.commonWords.grow().setTo(word,
						tmpDescWeights.get(wordIdx), invertedFile.wordWeights.get(invertedIdx));
			}
		}
	}

	/**
	 * List of images which observed a single word
	 */
	public static class InvertedFile {
		// image indexes
		public final DogArray_I32 images = new DogArray_I32();
		// TF-IDF descriptor weight for the word in the specified image
		public final DogArray_F32 wordWeights = new DogArray_F32();

		public void addImage( int imageIdx, float wieght ) {
			images.add(imageIdx);
			wordWeights.add(wieght);
		}

		public void reset() {
			images.reset();
			wordWeights.reset();
		}
	}

	/**
	 * Information on each potential candidate match to the querry image
	 */
	public static class Candidate implements Comparable<Candidate> {
		// Initially stores the image index, but is then converted into the image ID for output
		public int identification;
		// The error between this image's descriptor and the query image
		public float error;
		/** All words which are common between image in DB and the query image */
		public final DogArray<CommonWords> commonWords = new DogArray<>(CommonWords::new);

		public void reset() {
			identification = -1;
			error = -1;
			commonWords.reset();
		}

		@Override public int compareTo( @NotNull Candidate o ) {
			return Float.compare(error, o.error);
		}
	}
}
