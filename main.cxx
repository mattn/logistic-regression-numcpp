#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:26451 6262 26812 6297 4244)
#endif

#include <NumCpp.hpp>

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#include <vector>
#include <map>
#include <string>
#include <iostream>

static float
softmax(nc::NdArray<float>& w, nc::NdArray<float>& x) {
  auto v = nc::dot<float>(w, x).item();
  return 1.0f / (1.0f + std::exp(-v));
}

static float
predict(nc::NdArray<float>& w, nc::NdArray<float>& x) {
  return softmax(w, x);
}

static nc::NdArray<float>
logistic_regression(nc::NdArray<float>& X, nc::NdArray<float>& y, float rate, int ntrains) {
  auto w = nc::random::randN<float>({1, X.numCols()});

  for (auto n = 0; n < ntrains; n++) {
    for (nc::uint32 i = 0; i < X.numRows(); i++) {
      auto&& x = X.row(i);
      auto&& t = nc::NdArray<float>(x.copy());
      auto pred = softmax(t, w);
      auto perr = y.at(0, i) - pred;
      auto scale = rate * perr * pred * (1 - pred);
      auto&& dx = x.copy();
      dx += x;
      dx *= scale;
      w += dx;
    }
  }

  return w;
}

static std::vector<std::string>
split(std::string& fname, char delimiter) {
  std::istringstream f(fname);
  std::string field;
  std::vector<std::string> result;
  while (getline(f, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}

int main() {
  std::ifstream ifs("iris.csv");

  std::string line;

  // skip header
  std::getline(ifs, line);

  std::vector<float> rows;
  std::vector<std::string> names;
  while (std::getline(ifs, line)) {
    // sepal length, sepal width, petal length, petal width, name
    auto cells = split(line, ',');
    rows.push_back(std::stof(cells.at(0)));
    rows.push_back(std::stof(cells.at(1)));
    rows.push_back(std::stof(cells.at(2)));
    rows.push_back(std::stof(cells.at(3)));
    names.push_back(cells.at(4));
  }
  // make vector 4 dimentioned
  nc::NdArray<float> X(rows);
  X.reshape((nc::int32) rows.size()/4, 4);

  // make onehot values of names
  std::map<std::string, size_t> labels;
  for(auto& name : names) {
    if (labels.count(name) == 0) labels[name] = labels.size();
  }
  std::vector<float> counts;
  for (auto& name : names) {
    if (labels.count(name) > 0) counts.push_back((float)labels[name]);
  }
  nc::NdArray<float> y(counts);
  y /= (float) labels.size();

  names.clear();
  for(auto& k : labels) {
    names.push_back(k.first);
  }

  // make factor from input values
  auto w = logistic_regression(X, y, 0.1f, 300);

  // predict samples
  for (nc::uint32 i = 0; i < X.numRows(); i++) {
    auto x = X.row(i);
    auto n = (size_t) (predict(w, x) * (float) labels.size() + 0.5);
    if (n > names.size() - 1) n = names.size() - 1;
    std::cout << names[n] << std::endl;
  }

  return 0;
}
