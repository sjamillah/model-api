import pickle
import pprint


def inspect_pickle_file(file_path):
    try:
        with open("mental_health_model.pkl", "rb") as file:
            # Try to load the pickle file
            loaded_data = pickle.load(file)

            print("=" * 50)
            print(f"Pickle File Contents: {file_path}")
            print("=" * 50)

            # Print type of loaded data
            print(f"Data Type: {type(loaded_data)}")

            # If it's a dictionary, print keys and a preview of values
            if isinstance(loaded_data, dict):
                print("\nDictionary Keys:")
                for key in loaded_data.keys():
                    print(f"- {key}")

                print("\nDetailed Inspection:")
                for key, value in loaded_data.items():
                    print(f"\n{key}:")
                    try:
                        print(f"Type: {type(value)}")
                        # If it's a small/simple object, print its representation
                        if hasattr(value, "__len__"):
                            print(f"Length: {len(value)}")

                        # For smaller objects, print a preview
                        if isinstance(value, (list, dict, tuple)) and len(value) > 0:
                            print("Preview:")
                            pprint.pprint(value[:5] if len(value) > 5 else value)
                        elif not isinstance(value, (type, type(None))):
                            print("Value:", repr(value))
                    except Exception as e:
                        print(f"Could not fully inspect {key}: {e}")

            # If it's not a dictionary, print a more general representation
            else:
                print("\nFull Object Representation:")
                pprint.pprint(loaded_data)

    except Exception as e:
        print(f"Error loading pickle file: {e}")
        import traceback

        traceback.print_exc()


# Prompt user to enter the pickle file path
pickle_file_path = input("Enter the full path to your pickle file: ")
inspect_pickle_file(pickle_file_path)
