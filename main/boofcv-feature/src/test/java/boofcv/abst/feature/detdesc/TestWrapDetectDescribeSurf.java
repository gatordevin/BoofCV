/*
 * Copyright (c) 2011-2020, Peter Abeles. All Rights Reserved.
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

package boofcv.abst.feature.detdesc;

import boofcv.concurrency.BoofConcurrency;
import boofcv.factory.feature.detdesc.FactoryDetectDescribe;
import boofcv.struct.feature.BrightFeature;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.ImageType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * @author Peter Abeles
 */
public class TestWrapDetectDescribeSurf extends GenericTestsDetectDescribePoint<GrayF32,BrightFeature>
{

	static {
		BoofConcurrency.USE_CONCURRENT = false;
	}

	TestWrapDetectDescribeSurf() {
		super(true, true, ImageType.single(GrayF32.class), BrightFeature.class);
	}

	@Override
	public DetectDescribePoint<GrayF32, BrightFeature> createDetDesc() {
		return FactoryDetectDescribe.surfStable(null,null,null, GrayF32.class);
	}

	/**
	 * More rigorous test to see if sets is done correctly specific to SURF
	 */
	@Test
	void setsRigorous() {
		fail("implement");
	}
}
