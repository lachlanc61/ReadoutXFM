package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
)

/*
usage example:
go run conv.go -input=./nbb319_2.GeoPIXE -output=./temp.csv -detector=A
*/

/*
func printVal(v byte, newLine bool) {
    nl := "\t"
    if newLine {
        nl = "\n"
    }

    s := "_"
    if v > 32 && v < 255 {
        s = string(v)
    }

    fmt.Printf("0x%x (%v)\t%v%v", v, v, s, nl)
}
*/
const pixHeaderLength = 16

func readPixelHeader(pixelNum int, byteOffset int, bytes []byte) uint32 {
	if len(bytes) < pixHeaderLength {
		log.Fatalf("Pixel header: %v at byte offset %v was not long enough to be valid", pixelNum, byteOffset)
	}

	// First 2 bytes should be "DP"
	if bytes[0] != 'D' || bytes[1] != 'P' {
		log.Fatalf("Expected DP header for pixel %v at byte offset %v", pixelNum, byteOffset)
	}

	// Print it out as bytes first
	/*fmt.Println("Header as bytes:")
	  for _, b := range bytes {
	      printVal(b, false)
	  }
	*/
	pixelDataSize := binary.LittleEndian.Uint32(bytes[2:6])

	// We don't know what the other fields are, so ignore...
	/*
	   fmt.Printf("\ni32=%v i16=%v i16=%v i16=%v f32=%v\n",
	       pixelDataSize,
	       binary.LittleEndian.Uint16(bytes[4:6]),
	       binary.LittleEndian.Uint16(bytes[6:8]),
	       binary.LittleEndian.Uint16(bytes[8:10]),
	       float32(binary.LittleEndian.Uint32(bytes[10:14])),
	   )
	*/
	return pixelDataSize
}

func main() {
	var geopixeFile string
	var outputFile string
	var detectorLetter string

	flag.StringVar(&geopixeFile, "input", "/mnt/f/DATA/XFM/geopixe_conversion/test.geopixe", "Input file")
	flag.StringVar(&outputFile, "output", "", "Output file")
	flag.StringVar(&detectorLetter, "detector", "", "Detector letter")
	flag.Parse()

	fmt.Println("in")
	fmt.Println(&geopixeFile)
	fmt.Println("out")
	fmt.Println(&outputFile)
	fmt.Println("det")
	fmt.Println(&detectorLetter)

	if len(geopixeFile) <= 0 {
		log.Fatalln("No input file path specified")
	}

	if len(outputFile) <= 0 {
		log.Fatalln("No output file path specified")
	}

	if detectorLetter != "A" && detectorLetter != "B" {
		log.Fatalln("Invalid detector, expected A or B")
	}

	bytes, err := os.ReadFile(geopixeFile)
	if err != nil {
		log.Fatal(err)
	}

	outFile, err := os.Create(outputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer outFile.Close()

	fmt.Printf("File is %v bytes long\n", len(bytes))

	//perline := 4

	// File format seems to be:
	// 2 bytes for length of header
	// <header as JSON string>
	// Repeating {
	//     "pixels" starting with header "DP"
	//     14 bytes of header data
	//     Repeating uint16 of for channel#, count
	// }
	dataStart := int(binary.LittleEndian.Uint16(bytes[0:2])) + 2

	if dataStart == 20550 {
		// This is 0x5044 +2 (from above), which is "DP", meaning we found data straight away
		// so there is no file header!
		fmt.Println("No file header found!")
		dataStart = 0
	} else {
		fmt.Printf("File Header: %v\n", string(bytes[2:dataStart]))
	}

	const channelCount = 4096 // This should be read from the header JSON string

	// Write the header string
	csvLine := "PMC"
	for c := 0; c < channelCount; c++ {
		csvLine += fmt.Sprintf(",%v_%v", detectorLetter, c)
	}
	csvLine += "\n"
	outFile.WriteString(csvLine)

	pix := 0
	for c := dataStart; c < len(bytes)-pixHeaderLength; {
		fmt.Printf("Pixel: %v\n", pix)
		pixelDataSize := int(readPixelHeader(pix, c, bytes[c:c+pixHeaderLength]))

		// The rest should just be idx:uint16, count:uint16 pairs
		var counts [channelCount]int

		pixelDataEnd := c + pixelDataSize
		if pixelDataEnd > len(bytes) {
			fmt.Printf("Last pixel %v was saved containing %v bytes, but file truncated at %v bytes too early. Reading to end of file...\n", pix, pixelDataSize, pixelDataEnd-len(bytes))
			pixelDataEnd = len(bytes)
		}

		for i := c + pixHeaderLength; i+4 < pixelDataEnd; {
			channel := binary.LittleEndian.Uint16(bytes[i : i+2])
			count := binary.LittleEndian.Uint16(bytes[i+2 : i+4])

			//fmt.Printf("[%v]=%v\n", channel, count)
			if channel >= channelCount {
				log.Fatalf("Invalid channel %v at byte %v", channel, c)
			}

			counts[channel] = int(count)
			i += 4
		}

		// Write to output file
		csvLine = fmt.Sprintf("%v", pix)
		for _, count := range counts {
			csvLine += fmt.Sprintf(",%v", count)
		}
		csvLine += "\n"
		outFile.WriteString(csvLine)

		c += pixelDataSize

		pix++
	}

	fmt.Printf("Saved %v\n", outputFile)
}
