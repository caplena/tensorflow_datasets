# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Amazon Customer Reviews Dataset --- US REVIEWS DATASET."""

import collections
import json
import tensorflow.compat.v2 as tf

import tensorflow_datasets.public_api as tfds

_CITATION = """\
"""

_DESCRIPTION = """\
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns : 
    marketplace       - 2 letter country code of the marketplace where the review was written.
    customer_id       - Random identifier that can be used to aggregate reviews written by a single author.
    review_id         - The unique ID of the review.
    product_id        - The unique Product ID the review pertains to. In the multilingual dataset the reviews
                                            for the same product in different countries can be grouped by the same product_id.
    product_parent    - Random identifier that can be used to aggregate reviews for the same product.
    product_title     - Title of the product.
    product_category  - Broad product category that can be used to group reviews 
                                            (also used to group the dataset into coherent parts).
    star_rating       - The 1-5 star rating of the review.
    helpful_votes     - Number of helpful votes.
    total_votes       - Number of total votes the review received.
    vine              - Review was written as part of the Vine program.
    verified_purchase - The review is on a verified purchase.
    review_headline   - The title of the review.
    review_body       - The review text.
    review_date       - The date the review was written.
"""

_DATA_OPTIONS_V1_00 = [""]

_DATA_OPTIONS_LANGUAGES = ["de", "fr", "es", "en"]

_DATA_OPTIONS_SPLITS = ["dev", "test", "train"]

_DL_URL = "https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/{}/dataset_{}_{}.json"


class AmazonMultilingReviewsConfig(tfds.core.BuilderConfig):
  """BuilderConfig for AmazonUSReviews."""

  def __init__(self, *, data=None, **kwargs):
    """Constructs a AmazonMultilingReviews. """

    super(AmazonMultilingReviewsConfig, self).__init__(**kwargs)
    self.data = data


class AmazonMultilingReviews(tfds.core.GeneratorBasedBuilder):
  """AmazonMultilingReviews dataset."""

  BUILDER_CONFIGS = [AmazonMultilingReviewsConfig(  # pylint: disable=g-complex-comprehension
          name="reviews",
          description="A dataset consisting of multilingual reviews of Amazon products. Generate a split for each language in {}".format(', '.join(_DATA_OPTIONS_LANGUAGES)),
          version="0.1.0"
  )]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(
            {
                "review_id": tf.string,
                "product_id": tf.string,
                "reviewer_id": tf.string,
                "star_rating": tf.int32,
                "review_body": tf.string
            }
        ),
        supervised_keys=None,
        homepage="https://docs.opendata.aws/amazon-reviews-ml/readme.html",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    tfds_splits = []
    for lang in _DATA_OPTIONS_LANGUAGES:
      split_urls = {
          split: _DL_URL.format(split, lang, split)
          for split in _DATA_OPTIONS_SPLITS
      }
      split_paths = {
          split: dl_manager.download_and_extract(url)
          for split, url in split_urls.items()
      }

      tfds_splits.extend(
          [
              tfds.core.SplitGenerator(
                  name="{}{}".format(
                      lang, "_validate" if split == "test" else ""
                  ),
                  gen_kwargs={
                      "file_path": path,
                  }
              ) for split, path in split_paths.items()
          ]
      )

    # There is no predefined train/val/test split for this dataset.
    return tfds_splits

  def _generate_examples(self, file_path):
    """Generate features given the directory path.

            Args:
                file_path: path where the tsv file is stored

            Yields:
                The features.
            """

    with tf.io.gfile.GFile(file_path) as tsvfile:
      # Need to disable quoting - as dataset contains invalid double quotes.

      with open(file_path, 'r') as fh:
        for i, row in enumerate(fh):
          data = json.loads(row)

          yield i, {
              "review_id": data["review_id"],
              "product_id": data["product_id"],
              "reviewer_id": data["reviewer_id"],
              "star_rating": int(data["stars"]),
              "review_body": data["review_body"]
          }
