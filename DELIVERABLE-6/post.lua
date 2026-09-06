wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

local counter = 1
local samples = {
  '{"sno":1,"age":65,"gender":1,"cp":3,"trestbps":140,"chol":220,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":2}',
  '{"sno":2,"age":52,"gender":0,"cp":2,"trestbps":130,"chol":180,"fbs":1,"restecg":0,"thalach":160,"exang":1,"oldpeak":1.2,"slope":2,"ca":1,"thal":3}'
  -- add more samples here
}

request = function()
  local body = samples[counter]
  counter = counter + 1
  if counter > #samples then counter = 1 end
  return wrk.format(nil, nil, nil, body)
end
