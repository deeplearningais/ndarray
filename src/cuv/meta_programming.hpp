#if 0
#######################################################################################
# The MIT License

# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2012-2014  Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#endif
#ifndef __CUV_META_PROGRAMMING_HPP__
#define __CUV_META_PROGRAMMING_HPP__

namespace cuv {

/**
 * @addtogroup MetaProgramming
 * @{
 */

/// defines "False"
struct FalseType {
        enum {
            value = false
        };
};
/// defines "True"
struct TrueType {
        enum {
            value = true
        };
};

/**
 * @brief Checks whether two types are equal
 */
template<typename T1, typename T2>
struct IsSame
{
        /// is true only if T1==T2
        typedef FalseType Result;
};

/**
 * @see IsSame
 */
template<typename T>
struct IsSame<T, T>
{
        /// T==T, therefore Result==TrueType
        typedef TrueType Result;
};

/**
 * @brief Checks whether two types are different
 */
template<typename T1, typename T2>
struct IsDifferent
{
        /// is true only if T1!=T2
        typedef TrueType Result;
};

/**
 * @see IsDifferent
 */
template<typename T>
struct IsDifferent<T, T>
{
        /// T==T, therefore Result==FalseType
        typedef FalseType Result;
};

/**
 * @brief Remove "const" from a type
 */
template<typename T>
struct unconst {
        /// no change
        typedef T type;
};

/**
 * @see unconst
 */
template<typename T>
struct unconst<const T> {
        /// T without the const
        typedef T type;
};

/**
 * @brief Switch result depending on Condition
 */
template<bool Condition, class Then, class Else>
struct If {
        /// assume condition is true
        typedef Then result;
};
/**
 * @see If
 */
template<class Then, class Else>
struct If<false, Then, Else> {
        /// condition is false
        typedef Else result;
};

/**
 * @brief enable-if controlled creation of SFINAE conditions
 */
template<bool B, class T = void>
struct EnableIfC {
        typedef T type; /// enabling succeeded :-)
};

/// @see EnableIfC
template<class T>
struct EnableIfC<false, T> {
};

/// @see EnableIfC
template<class Cond, class T = void>
struct EnableIf: public EnableIfC<Cond::value, T> {
};

/// @see EnableIfC
template<class Cond, class T = void>
struct DisableIf: public EnableIfC<!Cond::value, T> {
};

/**
 * @}
 */
}

#endif
