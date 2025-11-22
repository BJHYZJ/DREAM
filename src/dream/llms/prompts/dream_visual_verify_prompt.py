# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

DREAM_VISUAL_VERIFY_PROMPT = """
You will receive one image and a text query describing an object.
Respond in exactly two lines:
    Caption: <one short sentence explaining whether the queried object is visible>
    Answer: True   (if the object is present)
            False  (if the object is absent)

Example1:
    Input: 
        The object you need to verify is blue bottle
    Output:
        Caption:
            The blue bottle is in center of iamge, so the answer is True
        Answer:
            True

Example2:
    Input:
        The object you need to verify is red apple
    Output:
        Caption:
            No red apple appears anywhere in the scene, so the answer is False
        Answer:
            False

Example3:
    Input:
        The object you need to verify is black backpack
    Output:
        Caption:
            A black backpack is on the chair to the left side of the image, so the answer is True
        Answer:
            True
"""
