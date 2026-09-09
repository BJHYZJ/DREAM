DREAM_VISUAL_GROUNDING_PROMPT = f"""
        For object query I give, you need to find images that the object is shown. You should first caption each image and then make conclusion.

        Example #1:
            Input:
                The object you need to find is blue bottle.
            Output:
                Caption:
                    Image 1 is a red bottle. Image 2 is a blue mug. Image 3 is a blue bag. Image 4 is a blue bottle. Image 5 is a blue bottle
                Images: 
                    4, 5

        Example #2:
            Input:
                The object you need to find is orange cup.
            Output:
                Caption:
                    Image 1 is a orange fruit. Image 2 is a orange sofa. Image 3 is a blue cup.
                Images:
                    None

        Example #3:
            Input:
                The object you need to find is potato chip
            Output:
                Caption:
                    Image 1 is a sofa. Image 2 is a potato chip. Image 3 is a pepper. Image 4 is a laptop.
                Images:
                    1"""
