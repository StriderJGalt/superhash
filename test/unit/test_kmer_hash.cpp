/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    test_kmer_hash.cpp
 * @ingroup
 * @author  tpan
 * @brief
 * @details
 *
 */



#include "utils/logging.h"

// include google test
#include <gtest/gtest.h>
#include "index/kmer_hash.hpp"
#include "kmerhash/hash.hpp"

#include <random>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <set>
#include <cmath>

#include "common/kmer.hpp"
#include "common/alphabets.hpp"
#include "common/alphabet_traits.hpp"



// include files to test

//TESTS: Hash functions.  test boundary cases - the number of unique outputs should be relatively close to number of unique inputs.

template <typename T>
class KmerHashTest : public ::testing::Test {
  protected:

    static std::vector<T> kmers;
    static std::set<T> unique_kmers;   // using set, since we are testing hash functions.

    static constexpr size_t iterations = 100000;

  public:
    static void SetUpTestCase()
    {
        T kmer;

      srand(0);
      for (unsigned int i = 0; i < T::size; ++i) {

          kmer.nextFromChar(rand() % T::KmerAlphabet::SIZE);
      }

      kmers.resize(iterations);
      for (size_t i = 0; i < iterations; ++i) {
    	  kmers[i] = kmer;
    	  unique_kmers.emplace(kmer);
          kmer.nextFromChar(rand() % T::KmerAlphabet::SIZE);
      }
    }

    static void TearDownTestCase() {
    	unique_kmers.clear();
    	std::vector<T>().swap(kmers);
    }

  protected:
    template <template <typename, bool> class H>
    void hash_vector64(std::string name) {
    	std::unordered_set<size_t> hashes;

       H<T, false> op;

      bool same = true;

      for (size_t i = 0; i < this->iterations; ++i) {
        hashes.emplace(op(this->kmers[i]));
      }

      //double limit = this->iterations;
      //double max_uniq_kmers = std::pow(static_cast<double>(T::KmerAlphabet::SIZE), static_cast<double>(T::size));
      //limit = (limit < max_uniq_kmers) ? limit : max_uniq_kmers;

//      same = (limit - hashes.size()) < (limit * 0.2);
      same = (hashes.size() == this->unique_kmers.size());
      //printf(" hash size: %lu, unique_kmers %lu, maxUnique %f, iterations %lu\n", hashes.size(), this->unique_kmers.size(), max_uniq_kmers, this->iterations);
      if (!same)
    	  BL_DEBUGF("ERROR: hash %s unique hashes %lu is not same as unique element count: %lu.  input size %lu",
    	            name.c_str(), hashes.size(), this->unique_kmers.size(), this->kmers.size());

      ASSERT_TRUE(same);

    }
    template <template <typename> class B, template <typename> class H>
    void hash_vector_vs_sse(std::string name) {

        B<T> bop;
       H<T> op;

       std::vector<uint32_t> truth(this->iterations, 0);
       std::vector<uint32_t> test(this->iterations, 0);

       for (size_t i = 0; i < this->iterations; ++i) {
         truth[i] = bop(this->kmers[i]);
         test[i] = op(this->kmers[i]);
       }

      bool same = true;

      for (size_t i = 0; i < this->iterations; ++i) {
        same &= (truth[i] == test[i]);

        if (truth[i] != test[i]) {
           std::cout << " iteration " << i << " kmer " << this->kmers[i] << std::endl;
        }


        ASSERT_EQ(truth[i], test[i]);
      }

      ASSERT_TRUE(same);

    }
    template <template <typename> class B, template <typename> class H>
    void hash_vector_vs_sse_batch(std::string name) {

        B<T> bop;
       H<T> op;

       std::vector<uint32_t> truth(this->iterations, 0);
       std::vector<uint32_t> test(this->iterations, 0);

       for (size_t i = 0; i < this->iterations; ++i) {
         truth[i] = bop(this->kmers[i]);
       }
       op.hash(this->kmers.data(), this->iterations, test.data());


      bool same = true;

      for (size_t i = 0; i < this->iterations; ++i) {
        same &= (truth[i] == test[i]);

        if (truth[i] != test[i]) {
           std::cout << " iteration " << i << " kmer " << this->kmers[i] << std::endl;
        }


        ASSERT_EQ(truth[i], test[i]);
      }

      ASSERT_TRUE(same);

      op.hash(this->kmers.data(), this->iterations-1, test.data());

     same = true;

     for (size_t i = 0; i < this->iterations - 1; ++i) {
       same &= (truth[i] == test[i]);
     }

     ASSERT_TRUE(same);



     op.hash(this->kmers.data(), this->iterations-2, test.data());

    same = true;

    for (size_t i = 0; i < this->iterations - 2; ++i) {
      same &= (truth[i] == test[i]);
    }

    ASSERT_TRUE(same);


    op.hash(this->kmers.data(), this->iterations-3, test.data());

   same = true;

   for (size_t i = 0; i < this->iterations - 3; ++i) {
     same &= (truth[i] == test[i]);
   }

   ASSERT_TRUE(same);


    }


};

template <typename T>
constexpr size_t KmerHashTest<T>::iterations;

template <typename T>
std::vector<T> KmerHashTest<T>::kmers;
template <typename T>
std::set<T> KmerHashTest<T>::unique_kmers;


// indicate this is a typed test
TYPED_TEST_CASE_P(KmerHashTest);

TYPED_TEST_P(KmerHashTest, stdcpp)
{
	this->template hash_vector64<bliss::kmer::hash::cpp_std >(std::string("cpp_std"));
}

TYPED_TEST_P(KmerHashTest, iden)
{
	this->template hash_vector64<bliss::kmer::hash::identity>(std::string("identity"));
}

TYPED_TEST_P(KmerHashTest, murmur)
{
	this->template hash_vector64<bliss::kmer::hash::murmur  >(std::string("murmur"));
}

TYPED_TEST_P(KmerHashTest, farm)
{
	this->template hash_vector64<bliss::kmer::hash::farm    >(std::string("farm"));
}

TYPED_TEST_P(KmerHashTest, murmur32sse)
{
	this->template hash_vector_vs_sse<fsc::hash::murmur32, fsc::hash::murmur3sse32>(std::string("murmur3_32_vs_sse"));
}

TYPED_TEST_P(KmerHashTest, murmur32sse_batch)
{
  this->template hash_vector_vs_sse_batch<fsc::hash::murmur32, fsc::hash::murmur3sse32>(std::string("murmur3_32_vs_sse_batch"));
}




REGISTER_TYPED_TEST_CASE_P(KmerHashTest, stdcpp, iden, murmur, farm, murmur32sse, murmur32sse_batch);

//////////////////// RUN the tests with different types.

// max of 50 cases
typedef ::testing::Types<
    ::bliss::common::Kmer< 31, bliss::common::DNA,   uint64_t>,  // 1 word, not full
    ::bliss::common::Kmer< 32, bliss::common::DNA,   uint64_t>,  // 1 word, full
    ::bliss::common::Kmer< 64, bliss::common::DNA,   uint64_t>,  // 2 words, full
    ::bliss::common::Kmer< 80, bliss::common::DNA,   uint64_t>,  // 3 words, not full
    ::bliss::common::Kmer< 96, bliss::common::DNA,   uint64_t>,  // 3 words, full
    ::bliss::common::Kmer< 15, bliss::common::DNA,   uint32_t>,  // 1 word, not full
    ::bliss::common::Kmer< 16, bliss::common::DNA,   uint32_t>,  // 1 word, full
    ::bliss::common::Kmer< 32, bliss::common::DNA,   uint32_t>,  // 2 words, full
    ::bliss::common::Kmer< 40, bliss::common::DNA,   uint32_t>,  // 3 words, not full
    ::bliss::common::Kmer< 48, bliss::common::DNA,   uint32_t>,  // 3 words, full
    ::bliss::common::Kmer<  7, bliss::common::DNA,   uint16_t>,  // 1 word, not full
    ::bliss::common::Kmer<  8, bliss::common::DNA,   uint16_t>,  // 1 word, full
    ::bliss::common::Kmer<  9, bliss::common::DNA,   uint16_t>,  // 2 words, not full
    ::bliss::common::Kmer< 16, bliss::common::DNA,   uint16_t>,  // 2 words, full
    ::bliss::common::Kmer<  3, bliss::common::DNA,    uint8_t>,  // 1 word, not full
    ::bliss::common::Kmer<  4, bliss::common::DNA,    uint8_t>,  // 1 word, full
    ::bliss::common::Kmer<  5, bliss::common::DNA,    uint8_t>,  // 2 words, not full
    ::bliss::common::Kmer<  8, bliss::common::DNA,    uint8_t>,  // 2 words, full
    ::bliss::common::Kmer< 19, bliss::common::DNA,    uint8_t>,  // 5 words, not full
     ::bliss::common::Kmer< 39, bliss::common::DNA,    uint8_t>,  // 10 words, not full
      ::bliss::common::Kmer< 59, bliss::common::DNA,    uint8_t>,  // 15 words, not full
    ::bliss::common::Kmer< 21, bliss::common::DNA5,  uint64_t>,  // 1 word, not full
    ::bliss::common::Kmer< 22, bliss::common::DNA5,  uint64_t>,  // 2 word, not full
    ::bliss::common::Kmer< 42, bliss::common::DNA5,  uint64_t>,  // 2 words, not full
    ::bliss::common::Kmer< 43, bliss::common::DNA5,  uint64_t>,  // 3 words, not full
    ::bliss::common::Kmer< 64, bliss::common::DNA5,  uint64_t>,  // 3 words, full
    ::bliss::common::Kmer<  2, bliss::common::DNA5,   uint8_t>,  // 1 word, not full
    ::bliss::common::Kmer<  3, bliss::common::DNA5,   uint8_t>,  // 2 word, not full
    ::bliss::common::Kmer<  5, bliss::common::DNA5,   uint8_t>,  // 2 words, not full
    ::bliss::common::Kmer<  6, bliss::common::DNA5,   uint8_t>,  // 3 words, not full
    ::bliss::common::Kmer<  8, bliss::common::DNA5,   uint8_t>,  // 3 words, full
    ::bliss::common::Kmer< 15, bliss::common::DNA16, uint64_t>,  // 1 word, not full
    ::bliss::common::Kmer< 16, bliss::common::DNA16, uint64_t>,  // 1 word, full
    ::bliss::common::Kmer< 32, bliss::common::DNA16, uint64_t>,  // 2 words, full
    ::bliss::common::Kmer< 40, bliss::common::DNA16, uint64_t>,  // 3 words, not full
    ::bliss::common::Kmer< 48, bliss::common::DNA16, uint64_t>,  // 3 words, full
    ::bliss::common::Kmer<  7, bliss::common::DNA16, uint32_t>,  // 1 word, not full
    ::bliss::common::Kmer<  8, bliss::common::DNA16, uint32_t>,  // 1 word, full
    ::bliss::common::Kmer< 16, bliss::common::DNA16, uint32_t>,  // 2 words, full
    ::bliss::common::Kmer< 20, bliss::common::DNA16, uint32_t>,  // 3 words, not full
    ::bliss::common::Kmer< 24, bliss::common::DNA16, uint32_t>,  // 3 words, full
    ::bliss::common::Kmer<  3, bliss::common::DNA16, uint16_t>,  // 1 word, not full
    ::bliss::common::Kmer<  4, bliss::common::DNA16, uint16_t>,  // 1 word, full
    ::bliss::common::Kmer<  5, bliss::common::DNA16, uint16_t>,  // 2 words, not full
    ::bliss::common::Kmer<  8, bliss::common::DNA16, uint16_t>,  // 2 words, full
    ::bliss::common::Kmer<  1, bliss::common::DNA16,  uint8_t>,  // 1 word, not full
    ::bliss::common::Kmer<  2, bliss::common::DNA16,  uint8_t>,  // 1 word, full
    ::bliss::common::Kmer<  3, bliss::common::DNA16,  uint8_t>,  // 2 words, not full
    ::bliss::common::Kmer<  4, bliss::common::DNA16,  uint8_t> //,   // 2 words, full
//    ::bliss::common::Kmer< 21, bliss::common::ASCII,  uint64_t>,  // 3 words, not full
//    ::bliss::common::Kmer< 21, bliss::common::ASCII,  uint8_t>,  // 3 words, not full
//    ::bliss::common::Kmer< 5120, bliss::common::ASCII,  uint64_t>  // 80 words - cpp_std should see collisions

> KmerHashTestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Bliss, KmerHashTest, KmerHashTestTypes);
