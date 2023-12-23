if __name__ == "__main__":
    import sys
    import os
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    if parent_directory != sys.path[-1]:
        sys.path.append(parent_directory)
        print("Append parent path : ",parent_directory)
    else :
        print("existing")
    for path in sys.path:
        print(path)

    from output_tool import basic_output_function, null_output_function, output_function1, acc_output_function, pearson_output_function

else :
    try:
        from .output_tool import basic_output_function, null_output_function, output_function1, acc_output_function, pearson_output_function
    except :
        from output_tool import basic_output_function, null_output_function, output_function1, acc_output_function, pearson_output_function

# from .output_tool import basic_output_function, null_output_function, output_function1, acc_output_function, pearson_output_function



def init_output_function(config):
    
    from output_tool import basic_output_function, null_output_function, output_function1, acc_output_function, pearson_output_function
    
    output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function,
    "out1": output_function1,
    "acc": acc_output_function,
    "pearson": pearson_output_function}

    
    name = config.get("output", "output_function")
    if name in output_function_dic:
        return output_function_dic[name]
    else:
        print("check config('output','output_function') <= [Basic,Null,out1,acc,pearson]")
        raise NotImplementedError
