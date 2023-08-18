def build_query(input_values, fields, booleans):
    query = {"$and": []}
    current_section = query["$and"]

    for i, (input_value, field, boolean) in enumerate(zip(input_values, fields, booleans)):
        if input_value:
            condition = {field: {"$regex": input_value, "$options": "i"}}

            if boolean == "AND":
                current_section.append(condition)
            elif boolean == "OR":
                current_section.append({"$or": [condition]})
            elif boolean == "AND NOT":
                current_section.append({"$nor": [condition]})
            elif boolean == "OR NOT":
                current_section.append({"$nor": [{"$or": [condition]}]})

    # Remove the top-level $and operator if it's an empty array or has only one element
    if len(query["$and"]) <= 1:
        query = query["$and"][0] if query["$and"] else {}

    return query

