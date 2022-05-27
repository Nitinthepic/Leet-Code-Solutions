
import java.util.*;

class Solution {
	public int maxArea(int[] height) {
		int leftSideIndex = 0;
		int rightSideIndex = height.length - 1;
		int kingArea = 0;

		while (leftSideIndex < height.length && rightSideIndex < height.length) {
			kingArea = Math.max(kingArea, areaCalc(height, leftSideIndex, rightSideIndex));
			if (leftSideIndex == rightSideIndex) break;
			if (height[leftSideIndex] <= height[rightSideIndex]) {
				leftSideIndex++;
			} else {
				rightSideIndex--;
			}

		}
		return kingArea;
	}

	private int areaCalc(int[] height, int leftSide, int rightSide) {

		return (rightSide - leftSide) * Math.min(height[leftSide], height[rightSide]);
	}

	public void rotate(int[][] matrix) {
		for (int[] row : matrix) {
			flipArray(row);
		}
		for (int i = 0; i < matrix.length - 1; i++) {
			for (int j = 0; j < matrix[i].length - 1; j++) {
				int temp = matrix[i][j];
				matrix[i][j] = matrix[matrix.length - 1 - j][matrix.length - 1 - i];
				matrix[matrix.length - 1 - j][matrix.length - 1 - i] = temp;
			}
		}
	}

	private void flipArray(int[] array) {
		int i = 0;
		int j = array.length - 1;
		while (i < j) {
			int temp = array[i];
			array[i] = array[j];
			array[j] = temp;
			i++;
			j--;
		}
	}

	public String longestPalindrome(String s) {
		boolean[][] palindromeCheck = new boolean[s.length()][s.length()];
		int start = 0;
		int end = 0;

		for (int i = s.length(); i >= 0; i--) {
			for (int j = i; j < s.length(); j++) {
				boolean b = s.charAt(i) == s.charAt(j);
				if (i == j) {
					palindromeCheck[i][j] = true;
				} else if (j - i == 1) {
					palindromeCheck[i][j] = b;
				} else if (b && palindromeCheck[i + 1][j - 1]) {
					palindromeCheck[i][j] = true;
				}
				if (palindromeCheck[i][j] && j - i >= end - start) {
					end = j;
					start = i;
				}
			}
		}
		return s.substring(start, end + 1);
	}

	public List<String> letterCombinations(String digits) {
		List<String> returnString = new ArrayList<>();
		if (digits.length() == 0) {
			return returnString;
		}
		returnString.add("");
		int index = 0;
		for (char digit : digits.toCharArray()) {
			int currentDigit = Integer.parseInt(String.valueOf(digit));
			while (returnString.get(0).length() == index) {
				String previousCombination = returnString.remove(0);
				for (char numToChar : convertIntToChar(currentDigit)) {
					System.out.println(previousCombination + numToChar);
					returnString.add(previousCombination + numToChar);
				}
			}
			index++;
		}
		return returnString;
	}

	public String longestCommonPrefix(String[] strs) {
		Arrays.sort(strs, Comparator.comparingInt(String::length));
		String returnValue = "";
		for (int i = 0; i < strs[0].length(); i++) {
			String currentChar = strs[0].substring(i, i + 1);
			System.out.println(currentChar);
			boolean charMatches = true;
			for (int j = 1; j < strs.length; j++) {
				if (!strs[j].substring(i, i + 1).equals(currentChar)) {
					charMatches = false;
					break;
				}
			}
			if (charMatches) {
				returnValue += currentChar;
			} else {
				break;
			}
		}
		return returnValue;
	}

	public List<List<Integer>> threeSum(int[] nums, int target) {
		Arrays.sort(nums);
		HashMap<Integer, Integer> valueSet = new HashMap<>();
		for (int i = 0; i < nums.length; i++) {
			valueSet.put(nums[i], i);
		}
		List<List<Integer>> returnValue = new ArrayList<>();
		for (int a = 0; a < nums.length - 3; a++) {
			for (int i = a; i < nums.length - 2; i++) {
				for (int j = i + 1; j < nums.length - 1; j++) {
					int remainder = target-(nums[i] + nums[j]+nums[a]);
					if (valueSet.containsKey(remainder) && valueSet.get(remainder) > j) {
						List<Integer> tempList = new ArrayList<>();
						tempList.add(nums[a]);
						tempList.add(nums[i]);
						tempList.add(nums[j]);
						tempList.add(remainder);
						returnValue.add(tempList);
					}
					j = valueSet.get(nums[j]);
				}
				i = valueSet.get(nums[i]);
			}
			a = valueSet.get(nums[a]);
		}
		return returnValue;
	}

	public boolean isValid(String s) {
		if (s.length() % 2 != 0) return false;
		Stack<Character> parenthesisStack = new Stack<>();
		for (char currentChar : s.toCharArray()) {
			switch (currentChar) {
				case '(':
					parenthesisStack.add(')');
					break;
				case '{':
					parenthesisStack.add('}');
					break;
				case '[':
					parenthesisStack.add(']');
					break;
				default:
					if (parenthesisStack.isEmpty()) return false;
					if (parenthesisStack.peek().equals(currentChar))
						parenthesisStack.pop();
					else return false;

			}
		}
		return parenthesisStack.size() == 0;
	}

	private void flipArray(int[] array, int i, int j) {
		while (i < j) {
			int temp = array[i];
			array[i] = array[j];
			array[j] = temp;
			i++;
			j--;
		}
	}

	public void nextPermutation(int[] nums) {
		if (!(nums.length < 2)) {
			int i = 0;
			while (i < nums.length - 1 && nums[i] < nums[i + 1]) {
				i++;
			}
			int pivot = nums[i - 1];
			int pivotSuccessor = 0;
			int pivotDifference = nums[i] - pivot;
			for (int j = i; j < nums.length; j++) {
				if (pivotDifference > nums[j] - pivot) {
					pivotDifference = nums[j] - pivot;
					pivotSuccessor = j;
				}
			}
			int temp = nums[i - 1];
			nums[i - 1] = nums[pivotSuccessor];
			nums[pivotSuccessor] = temp;
			flipArray(nums, i, nums.length - 1);
		}

	}

	public int findMaxForm(String[] strs, int m, int n) {
		//creates the 3D array for the knapsack
		int[][][] calcSumArray = new int[strs.length + 1][m + 1][n + 1];
		//traverses through all strings in strs where i = 1 corresponds to strs[0]
		for (int i = 0; i < strs.length + 1; i++) {
			int numberOfZeroes = 0;
			int numberOfOnes = 0;
			//avoids index out of bound
			if (i != 0) {
				for (char inChar : strs[i - 1].toCharArray()) {
					if (inChar == '0') numberOfZeroes++;
					if (inChar == '1') numberOfOnes++;
				}
			}
			System.out.println(numberOfZeroes);
			System.out.println(numberOfOnes);
			//traverses through all possible number of zeroes allowed less than or equal to m
			for (int j = 0; j < m + 1; j++) {
				//traverses through all possible numbers of ones allowed less than or equal to n
				for (int k = 0; k < n + 1; k++) {
					//if i == 0, we are choosing no items out strs, so value must be zero
					if (i == 0) calcSumArray[i][j][k] = 0;
					else if (numberOfZeroes <= j && numberOfOnes <= k) {
						//if we have the ability to choose the current String (strs[i-1]), choose it if and only if choosing it results in a greater value than not choosing it
						calcSumArray[i][j][k] = Math.max(calcSumArray[i - 1][j][k], calcSumArray[i - 1][j - numberOfZeroes][k - numberOfOnes] + 1);
					} else {
						//if the strings is just too heavy, its value is whatever the value is if that item was not even an option
						calcSumArray[i][j][k] = calcSumArray[i - 1][j][k];
					}
				}
			}
		}//return the apex value
		return calcSumArray[strs.length][m][n];
	}


	public int longestPalindromeSubseq(String s) {
		int[][] palindromeMatrix = new int[s.length()][s.length()];
		for (int i = s.length() - 1; i >= 0; i--) {
			//a letter by itself is palindrome of size 1
			palindromeMatrix[i][i] = 1;
			for (int j = i + 1; j < s.length(); j++) {
				if (s.charAt(i) == s.charAt(j))
					//if the substring ends/starts with same letter then keeping that letter adds two the string length, one on both sides
					palindromeMatrix[i][j] = palindromeMatrix[i + 1][j - 1] + 2;
				else
					//if not, then we need to see if keeping or removing that letter results in the longer palindrome, i+1 removes that letter, j-1 keeps that letter but removes the last letter of our word
					palindromeMatrix[i][j] = Math.max(palindromeMatrix[i + 1][j], palindromeMatrix[i][j - 1]);
			}
		}
		return palindromeMatrix[0][s.length() - 1];
	}

	public List<String> generateParenthesis(int n) {
		List<String> generatedParen = new ArrayList<>();
		helper(0, 0, n, "", generatedParen);
		return generatedParen;
	}

	private void helper(int openParen, int closeParen, int n, String cur, List<String> currentList) {
		if (cur.length() == n * 2) {
			currentList.add(cur);
		}
		if (openParen < n) {
			helper(openParen + 1, closeParen, n, cur + "(",
					currentList);
		}
		if (closeParen < openParen) {
			helper(openParen, closeParen + 1, n, cur + ")",
					currentList);
		}
	}

	private char[] convertIntToChar(int num) {
		char[][] characterMapping = {{}, {}, {'a', 'b', 'c'}, {'d', 'e', 'f'}, {'g', 'h', 'i'}, {'j', 'k', 'l'}, {'m', 'n', 'o'}, {'p', 'q', 'r', 's'}, {'t', 'u', 'v'}, {'w', 'x', 'y', 'z'}};
		return characterMapping[num];

	}

	public boolean canBeValid(String s, String locked) {
		if (s.length() % 2 != 0)
			return false;
		int comparison = 0;
		for (int i = 0; i < s.length(); i++) {
			if (locked.charAt(i) == '0' || s.charAt(0) == '(')
				comparison++;
			else comparison--;
			if (comparison < 0)
				return false;
		}
		for (int i = 0; i < s.length(); i++) {
			if (locked.charAt(i) == '0' || s.charAt(0) == ')')
				comparison++;
			else comparison--;
			if (comparison < 0)
				return false;
		}
		return true;
	}

	public String minRemoveToMakeValid(String s) {
		Stack<Character> validParenthesis = new Stack<>();
		Stack<Integer> invalidParenthesis = new Stack<>();
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				validParenthesis.add(')');
				invalidParenthesis.add(i);
			}
			if (s.charAt(i) == ')') {
				if (validParenthesis.isEmpty()) {
					validParenthesis.add('(');
					invalidParenthesis.add(i);
				} else if (validParenthesis.peek() == ')') {
					validParenthesis.pop();
					invalidParenthesis.pop();
				} else {
					validParenthesis.add('(');
					invalidParenthesis.add(i);
				}
			}
		}
		System.out.println(validParenthesis);
		System.out.println(invalidParenthesis);
		StringBuilder sb = new StringBuilder(s);
		while (!invalidParenthesis.isEmpty()) {
			sb.setCharAt(invalidParenthesis.pop(), ' ');
		}
		return (sb.toString()).replace(" ", "");
	}

	public int longestValidParentheses(String s) {
		Stack<Integer> invalidParenthesis = new Stack<>();
		int answer = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				invalidParenthesis.add(i);
			} else {
				if (invalidParenthesis.isEmpty()) {
					invalidParenthesis.add(i);
				} else if (s.charAt(invalidParenthesis.peek()) == '(') {
					invalidParenthesis.pop();
				} else {
					invalidParenthesis.add(i);
				}
			}
		}
		int start = 0;
		int end = s.length();
		if (invalidParenthesis.isEmpty())
			return s.length();
		while (!invalidParenthesis.isEmpty()) {
			start = invalidParenthesis.pop();
			answer = Math.max(answer, end - start - 1);
			end = start;
		}
		answer = Math.max(answer, end);
		return answer;
	}

	public int[] searchRange(int[] nums, int target) {
		return new int[]{lowestIndexBinarySearch(nums, target),
				highestIndexBinarySearch(nums, target)};
	}

	private int lowestIndexBinarySearch(int[] nums, int target) {
		int low = 0;
		int high = nums.length - 1;
		int i = -1;
		while (low <= high) {
			int mid = low + (high - low) / 2;
			if (nums[mid] >= target)
				high = mid - 1;
			else {
				low = mid + 1;
			}
			if (nums[mid] == target)
				i = mid;
		}
		return i;
	}

	private int highestIndexBinarySearch(int[] nums, int target) {
		int low = 0;
		int high = nums.length - 1;
		int i = -1;
		while (low <= high) {
			int mid = low + (high - low) / 2;
			if (nums[mid] <= target)
				low = mid + 1;
			else {
				high = mid - 1;
			}
			if (nums[mid] == target)
				i = mid;
		}
		return i;
	}

	public int lengthOfLIS(int[] nums) {
		ArrayList<Integer> subseqValues = new ArrayList<>();
		int index = 0;
		for (int num : nums) {
			if (index == 0 || subseqValues.get(subseqValues.size() - 1) < num) {
				index++;
				subseqValues.add(num);
			} else {
				System.out.println(subseqValues);
				subseqValues.set(BinarySearch(subseqValues, num), num);
			}
		}
		System.out.println(subseqValues);
		return index;
	}

	private int BinarySearch(ArrayList<Integer> nums, int target) {
		int low = 0;
		int high = nums.size() - 1;
		while (low < high) {
			int mid = low + (high - low) / 2;
			if (nums.get(mid) == target)
				return mid;
			if (nums.get(mid) > target)
				high = mid;
			else {
				low = mid + 1;
			}
		}
		return low;
	}

	public int maxEnvelopes(int[][] envelopes) {
		Arrays.sort(envelopes, (x, y) -> {
			if (x[0] - y[0] == 0) {
				return -(x[1] - y[1]);
			}
			return x[0] - y[0];
		});
		return lengthOfLISDoll(envelopes);
	}

	public int lengthOfLISDoll(int[][] nums) {
		ArrayList<Integer> subseqValues = new ArrayList<>();
		int index = 0;
		for (int i = 0; i < nums.length; i++) {
			if (index == 0 || subseqValues.get(subseqValues.size() - 1) < nums[i][1]) {
				index++;
				subseqValues.add(nums[i][1]);
			} else {
				System.out.println(subseqValues);
				subseqValues.set(BinarySearch(subseqValues, nums[i][1]),
						nums[i][1]);
			}
		}
		System.out.println(subseqValues);
		return index;
	}

	public int hammingWeight(int n) {
		int numOfOnes = 0;
		for (int i = 0; i < 32; i++) {
			numOfOnes += n & 1;
			n = n >>> 1;
		}
		return numOfOnes;
	}


	public int hammingDistance(int x, int y) {
		int numOfDif = 0;
		for (int i = 0; i < 32; i++) {
			if ((x & 1) != (y & 1))
				numOfDif++;
			x = x >>> 1;
			y = y >>> 1;
		}
		return numOfDif;
	}

	public int countWords(String[] words1, String[] words2) {
		HashMap<String, Integer> dog = new HashMap<>();
		HashMap<String, Integer> cat = new HashMap<>();
		int twoCount = 0;
		for (String word : words1) {
			if (dog.containsKey(word)) {
				int temp = dog.get(word);
				dog.replace(word,temp+1);
			} else {
				dog.put(word, 1);
			}
		}
		for(String word: words2){
			if (cat.containsKey(word)) {
				int temp = cat.get(word);
				cat.replace(word,temp+1);
			} else {
				cat.put(word, 1);
			}
		}
		for (Map.Entry<String,Integer> entry :dog.entrySet()) {
			if(entry.getValue()==1&&cat.containsKey(entry.getKey())&&cat.get(entry.getKey())==1)
				twoCount++;
		}
		return twoCount;
	}

	public int[] intersection(int[] nums1, int[] nums2) {
		Set<Integer> dog = new HashSet<>();
		Set<Integer> cat = new HashSet<>();

		for (int num: nums1) {
			dog.add(num);
		}
		for(int num: nums2){
			if(dog.contains(num))
				cat.add(num);
		}
		int[] ts = new int[cat.size()];
		int i = 0;
		for (int elem: cat) {
			ts[i] = elem;
			i++;
		}
		return ts;
	}

	public int[] intersect(int[] nums1, int[] nums2) {
		HashMap<Integer, Integer> nums1Track = new HashMap<>();
		HashMap<Integer, Integer> nums2Track = new HashMap<>();
		ArrayList<Integer> endResult = new ArrayList<>();
		for (int word : nums1) {
			if (nums1Track.containsKey(word)) {
				int temp = nums1Track.get(word);
				nums1Track.replace(word,temp+1);
			} else {
				nums1Track.put(word, 1);
			}
		}
		for(int word: nums2){
			if (nums2Track.containsKey(word)) {
				int temp = nums2Track.get(word);
				nums2Track.replace(word,temp+1);
			} else {
				nums2Track.put(word, 1);
			}
		}
		for (Map.Entry<Integer,Integer> entry :nums1Track.entrySet()) {
			if(nums2Track.containsKey(entry.getKey()))
				for(int i = 0; i < Math.min(entry.getValue(),nums2Track.get(entry.getKey()));i++){
					endResult.add(entry.getKey());
				}
		}
		int[] rv = new int[endResult.size()];
		for (int i = 0; i < rv.length; i++) {
			rv[i] = endResult.get(i);
		}
		return rv;
	}

	public int numberOfSteps(int num) {
		if(num==0)
			return 0;
		if((num&1)==0)
			return 1+numberOfSteps(num/2);
		else
			return 1+numberOfSteps(num-1);
	}

	public int countOperations(int num1, int num2) {
		int steps = 0;
		while(num1!=0&&num2!=0){
			if(num1<num2){
				num2 = num2 - num1;
			}else{
				num1 = num1 - num2;
			}
			steps++;
		}
		return steps;
	}


	public static void main(String[] args) {
		Solution dog = new Solution();
		String[] temp = {"1", "0001", "111001"};
		int[][] mouse = {{1, 3}, {3, 5}, {6, 7}, {6, 8}, {8, 4}, {9, 5}};
		System.out.println(dog.hammingDistance(0, (int) (Math.pow(2, 31) - 1)));

	}
}